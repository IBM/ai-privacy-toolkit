from abc import ABCMeta, abstractmethod
from typing import Any, Optional, Callable, Tuple, Union, TYPE_CHECKING
from enum import Enum, auto
import numpy as np

from apt.utils.datasets import Dataset, Data, OUTPUT_DATA_ARRAY_TYPE
from art.estimators.classification import BlackBoxClassifier
from art.utils import check_and_transform_label_format

if TYPE_CHECKING:
    import torch


class ModelOutputType(Enum):
    CLASSIFIER_PROBABILITIES = auto()  # vector of probabilities
    CLASSIFIER_LOGITS = auto()  # vector of logits
    CLASSIFIER_SCALAR = auto()  # label only
    REGRESSOR_SCALAR = auto()  # value


class ModelType(Enum):
    SKLEARN_DECISION_TREE = auto()
    SKLEARN_GRADIENT_BOOSTING = auto()


class ScoringMethod(Enum):
    ACCURACY = auto()  # number of correct predictions divided by the number of samples
    MEAN_SQUARED_ERROR = auto()  # mean squared error between the predictions and true labels


def is_one_hot(y: OUTPUT_DATA_ARRAY_TYPE) -> bool:
    return len(y.shape) == 2 and y.shape[1] > 1


def get_nb_classes(y: OUTPUT_DATA_ARRAY_TYPE) -> int:
    """
    Get the number of classes from an array of labels

    :param y: The labels
    :type y: numpy array
    :return: The number of classes as integer
    """
    if y is None:
        return 0

    if type(y) != np.ndarray:
        raise ValueError("Input should be numpy array")

    if is_one_hot(y):
        return y.shape[1]
    else:
        return int(np.max(y) + 1)


def check_correct_model_output(y: OUTPUT_DATA_ARRAY_TYPE, output_type: ModelOutputType):
    """
    Checks whether there is a mismatch between the declared model output type and its actual output.
    :param y: Model output
    :type y: numpy array
    :param output_type: Declared output type (provided at init)
    :type output_type: ModelOutputType
    :raises: ValueError (in case of mismatch)
    """
    if not is_one_hot(y):  # 1D array
        if output_type == ModelOutputType.CLASSIFIER_PROBABILITIES or output_type == ModelOutputType.CLASSIFIER_LOGITS:
            raise ValueError("Incompatible model output types. Model outputs 1D array of categorical scalars while "
                             "output type is set to ", output_type)


class Model(metaclass=ABCMeta):
    """
    Abstract base class for ML model wrappers.

    :param model: The original model object (of the underlying ML framework)
    :type model: framework-specific model object
    :param output_type: The type of output the model yields (vector/label only for classifiers,
                        value for regressors)
    :type output_type: `ModelOutputType`
    :param black_box_access: Boolean describing the type of deployment of the model (when in production).
                             Set to True if the model is only available via query (API) access, i.e.,
                             only the outputs of the model are exposed, and False if the model internals
                             are also available. Default is True.
    :type black_box_access: boolean, optional
    :param unlimited_queries: If black_box_access is True, this boolean indicates whether a user can perform
                              unlimited queries to the model API or whether there is a limit to the number of
                              queries that can be submitted. Default is True.
    :type unlimited_queries: boolean, optional
    """

    def __init__(self, model: Any, output_type: ModelOutputType, black_box_access: Optional[bool] = True,
                 unlimited_queries: Optional[bool] = True, **kwargs):
        self._model = model
        self._output_type = output_type
        self._black_box_access = black_box_access
        self._unlimited_queries = unlimited_queries

    @abstractmethod
    def fit(self, train_data: Dataset, **kwargs) -> None:
        """
        Fit the model using the training data.

        :param train_data: Training data.
        :type train_data: `Dataset`
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: Dataset, **kwargs) -> OUTPUT_DATA_ARRAY_TYPE:
        """
        Perform predictions using the model for input `x`.

        :param x: Input samples.
        :type x: `Dataset`
        :return: Predictions from the model as numpy array.
        """
        raise NotImplementedError

    @abstractmethod
    def score(self, test_data: Dataset, **kwargs):
        """
        Score the model using test data.

        :param test_data: Test data.
        :type train_data: `Dataset`
        :return: the score as float (for classifiers, between 0 and 1)
        """
        raise NotImplementedError

    @property
    def model(self) -> Any:
        """
        Return the underlying model.

        :return: The model.
        """
        return self._model

    @property
    def output_type(self) -> ModelOutputType:
        """
        Return the model's output type.

        :return: The model's output type.
        """
        return self._output_type

    @property
    def black_box_access(self) -> bool:
        """
        Return whether the model is only available via query (API) access, i.e.,
        only the outputs of the model are exposed, or if the model internals are also available.

        :return: True if the model is only available via query (API) access, otherwise False.
        """
        return self._black_box_access

    @property
    def unlimited_queries(self) -> bool:
        """
        If black_box_access is True, return whether a user can perform unlimited queries to the model API
        or whether there is a limit to the number of queries that can be submitted.

        :return: True if a user can perform unlimited queries to the model API, otherwise False.
        """
        return self._unlimited_queries


class BlackboxClassifier(Model):
    """
    Wrapper for black-box ML classification models.

    :param model: The training and/or test data along with the model's predictions for the data or a callable predict
                  method.
    :type model: `Data` object or Callable
    :param output_type: The type of output the model yields (vector/label only)
    :type output_type: `ModelOutputType`
    :param black_box_access: Boolean describing the type of deployment of the model (when in production).
                             Always assumed to be True (black box) for this wrapper.
    :type black_box_access: boolean, optional
    :param unlimited_queries: Boolean indicating whether a user can perform unlimited queries to the model API.
    :type unlimited_queries: boolean, optional
    :param model_type: The type of model this BlackboxClassifier represents. Needed in order to build and/or fit
                       similar dummy/shadow models.
    :type model_type: Either a (unfitted) model object of the underlying framework, or a ModelType representing the
                      type of the model, optional.
    :param loss: For pytorch models, the loss function used for training. Needed in order to build and/or fit
                 similar dummy/shadow models.
    :type loss: torch.nn.modules.loss._Loss, optional
    :param optimizer: For pytorch models, the optimizer used for training. Needed in order to build and/or fit
                      similar dummy/shadow models.
    :type optimizer: torch.optim.Optimizer, optional
    """
    def __init__(self, model: Any, output_type: ModelOutputType, black_box_access: Optional[bool] = True,
                 unlimited_queries: Optional[bool] = True, model_type: Optional[Union[Any, ModelType]] = None,
                 loss: "torch.nn.modules.loss._Loss" = None, optimizer: "torch.optim.Optimizer" = None,
                 **kwargs):
        super().__init__(model, output_type, black_box_access=True, unlimited_queries=unlimited_queries, **kwargs)
        self._nb_classes = None
        self._input_shape = None
        self._model_type = model_type
        self._loss = loss
        self._optimizer = optimizer

    @property
    def nb_classes(self) -> int:
        """
        Return the number of prediction classes of the model.

        :return: Number of prediction classes of the model.
        """
        return self._nb_classes

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """
        Return the shape of input to the model.

        :return: Shape of input to the model.
        """
        return self._input_shape

    @property
    def model_type(self) -> Optional[Union[Any, ModelType]]:
        """
        Return the type of the model.

        :return: Either a (unfitted) model object of the underlying framework, or a ModelType representing the type of
                 the model, or None (of none provided at init).
        """
        return self._model_type

    @property
    def loss(self):
        """
        The pytorch model's loss function.

        :return: The pytorch model's loss function.
        """
        return self._loss

    @property
    def optimizer(self):
        """
        The pytorch model's optimizer.

        :return: The pytorch model's optimizer.
        """
        return self._optimizer

    def fit(self, train_data: Dataset, **kwargs) -> None:
        """
        A blackbox model cannot be fit.
        """
        raise NotImplementedError

    def predict(self, x: Dataset, **kwargs) -> OUTPUT_DATA_ARRAY_TYPE:
        """
        Get predictions from the model for input `x`. `x` must be a subset of the data provided in the `model` data in
        `__init__()`.

        :param x: Input samples.
        :type x: `Dataset`
        :return: Predictions from the model as numpy array.
        """
        predictions = self._art_model.predict(x.get_samples())
        check_correct_model_output(predictions, self.output_type)
        return predictions

    def score(self, test_data: Dataset, scoring_method: Optional[ScoringMethod] = ScoringMethod.ACCURACY, **kwargs):
        """
        Score the model using test data.

        :param test_data: Test data.
        :type train_data: `Dataset`
        :param scoring_method: The method for scoring predictions. Default is ACCURACY.
        :type scoring_method: `ScoringMethod`, optional
        :return: the score as float (for classifiers, between 0 and 1)
        """
        if test_data.get_samples() is None or test_data.get_labels() is None:
            raise ValueError('score can only be computed when test data and labels are available')
        predicted = self._art_model.predict(test_data.get_samples())
        y = check_and_transform_label_format(test_data.get_labels(), nb_classes=self._nb_classes)
        if scoring_method == ScoringMethod.ACCURACY:
            return np.count_nonzero(np.argmax(y, axis=1) == np.argmax(predicted, axis=1)) / predicted.shape[0]
        else:
            raise NotImplementedError

    @abstractmethod
    def get_predictions(self) -> Union[Callable, Tuple[OUTPUT_DATA_ARRAY_TYPE, OUTPUT_DATA_ARRAY_TYPE]]:
        """
        Return all the data for which the model contains predictions, or the predict function of the model.

        :return: Tuple containing data and predictions as numpy arrays or callable.
        """
        raise NotImplementedError


class BlackboxClassifierPredictions(BlackboxClassifier):
    """
    Wrapper for black-box ML classification models using data and predictions.

    :param model: The training and/or test data along with the model's predictions for the data. Assumes that the data
                  is represented as numpy arrays. Labels are expected to either be class probabilities (multi-column) or
                  a 1D-array of categorical labels (consecutive integers starting at 0).
    :type model: `Data` object
    :param output_type: The type of output the model yields (vector/label only)
    :type output_type: `ModelOutputType`
    :param black_box_access: Boolean describing the type of deployment of the model (when in production).
                             Always assumed to be True for this wrapper.
    :type black_box_access: boolean, optional
    :param unlimited_queries: Boolean indicating whether a user can perform unlimited queries to the model API.
                              Always assumed to be False for this wrapper.
    :type unlimited_queries: boolean, optional
    """

    def __init__(self, model: Data, output_type: ModelOutputType, black_box_access: Optional[bool] = True,
                 unlimited_queries: Optional[bool] = True, **kwargs):
        super().__init__(model, output_type, black_box_access=True, unlimited_queries=False, **kwargs)
        x_train_pred = model.get_train_samples()
        y_train_pred = model.get_train_predictions()
        if y_train_pred is None:
            y_train_pred = model.get_train_labels()
        x_test_pred = model.get_test_samples()
        y_test_pred = model.get_test_predictions()
        if y_test_pred is None:
            y_test_pred = model.get_test_labels()

        if y_train_pred is not None:
            check_correct_model_output(y_train_pred, self.output_type)
        if y_test_pred is not None:
            check_correct_model_output(y_test_pred, self.output_type)

        if y_train_pred is not None and len(y_train_pred.shape) == 1:
            self._nb_classes = get_nb_classes(y_train_pred)
            y_train_pred = check_and_transform_label_format(y_train_pred, nb_classes=self._nb_classes)
        if y_test_pred is not None and len(y_test_pred.shape) == 1:
            if self._nb_classes is None:
                self._nb_classes = get_nb_classes(y_test_pred)
            y_test_pred = check_and_transform_label_format(y_test_pred, nb_classes=self._nb_classes)

        if x_train_pred is not None and y_train_pred is not None and x_test_pred is not None and y_test_pred is not None:
            if type(y_train_pred) != np.ndarray or type(y_test_pred) != np.ndarray \
               or type(y_train_pred) != np.ndarray or type(y_test_pred) != np.ndarray:
                raise NotImplementedError("X/Y Data should be numpy array")
            x_pred = np.vstack((x_train_pred, x_test_pred))
            y_pred = np.vstack((y_train_pred, y_test_pred))
        elif x_test_pred is not None and y_test_pred is not None:
            x_pred = x_test_pred
            y_pred = y_test_pred
        elif x_train_pred is not None and y_train_pred is not None:
            x_pred = x_train_pred
            y_pred = y_train_pred
        else:
            raise NotImplementedError("Invalid data - None")

        self._nb_classes = get_nb_classes(y_pred)
        self._input_shape = x_pred.shape[1:]
        self._x_pred = x_pred
        self._y_pred = y_pred
        predict_fn = (x_pred, y_pred)
        self._art_model = BlackBoxClassifier(predict_fn, self._input_shape, self._nb_classes, fuzzy_float_compare=True,
                                             preprocessing=None)

    def get_predictions(self) -> Union[Callable, Tuple[OUTPUT_DATA_ARRAY_TYPE, OUTPUT_DATA_ARRAY_TYPE]]:
        """
        Return all the data for which the model contains predictions.

        :return: Tuple containing data and predictions as numpy arrays.
        """
        return self._x_pred, self._y_pred


class BlackboxClassifierPredictFunction(BlackboxClassifier):
    """
    Wrapper for black-box ML classification models using a predict function.

    :param model: Function that takes in an `np.ndarray` of input data and returns predictions either as class
                  probabilities (multi-column) or a 1D-array of categorical labels (consecutive integers starting at 0).
    :type model: Callable
    :param output_type: The type of output the model yields (vector/label only)
    :type output_type: `ModelOutputType`
    :param input_shape: Shape of input to the model.
    :type input_shape: Tuple[int, ...]
    :param nb_classes: Number of prediction classes of the model.
    :type  nb_classes: int
    :param black_box_access: Boolean describing the type of deployment of the model (when in production).
                             Always assumed to be True for this wrapper.
    :type black_box_access: boolean, optional
    :param unlimited_queries: Boolean indicating whether a user can perform unlimited queries to the model API.
    :type unlimited_queries: boolean, optional
    """

    def __init__(self, model: Callable, output_type: ModelOutputType, input_shape: Tuple[int, ...], nb_classes: int,
                 black_box_access: Optional[bool] = True, unlimited_queries: Optional[bool] = True, **kwargs):
        super().__init__(model, output_type, black_box_access=True, unlimited_queries=unlimited_queries, **kwargs)
        self._nb_classes = nb_classes
        self._input_shape = input_shape

        def predict_wrapper(x):
            predictions = self.model(x)
            if not is_one_hot(predictions):
                predictions = check_and_transform_label_format(predictions, nb_classes=nb_classes, return_one_hot=True)
            return predictions

        self._predict_fn = predict_wrapper
        self._art_model = BlackBoxClassifier(predict_wrapper, self._input_shape, self._nb_classes, preprocessing=None)

    def get_predictions(self) -> Union[Callable, Tuple[OUTPUT_DATA_ARRAY_TYPE, OUTPUT_DATA_ARRAY_TYPE]]:
        """
        Return the predict function of the model.

        :return: Callable representing a function that takes in an `np.ndarray` of input data and returns predictions
                 either as class probabilities (multi-column) or a 1D-array of categorical labels (consecutive
                 integers starting at 0).
        """
        return self._predict_fn
