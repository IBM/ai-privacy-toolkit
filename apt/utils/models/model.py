from abc import ABCMeta, abstractmethod
from typing import Any, Optional
from enum import Enum, auto
import numpy as np

from apt.utils.datasets import Dataset, Data, OUTPUT_DATA_ARRAY_TYPE
from art.estimators.classification import BlackBoxClassifier
from art.utils import check_and_transform_label_format


class ModelOutputType(Enum):
    CLASSIFIER_PROBABILITIES = auto()  # vector of probabilities
    CLASSIFIER_LOGITS = auto()  # vector of logits
    CLASSIFIER_SCALAR = auto()  # label only
    REGRESSOR_SCALAR = auto()  # value


class ScoringMethod(Enum):
    ACCURACY = auto()  # number of correct predictions divided by the number of samples
    MEAN_SQUARED_ERROR = auto()  # mean squared error between the predictions and true labels


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
        return NotImplementedError

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

    def get_nb_classes(self, y: OUTPUT_DATA_ARRAY_TYPE) -> int:
        """
        Get the number of classes from an array of labels

        :param y: the labels
        :type y: numpy array
        :return: the number of classes as integer
        """
        if len(y.shape) == 1:
            return np.max(y) + 1
        else:
            return y.shape[1]


class BlackboxClassifier(Model):
    """
    Wrapper for black-box ML classification models.

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
        self.nb_classes = None
        x_train_pred = model.get_train_samples()
        y_train_pred = model.get_train_labels()
        x_test_pred = model.get_test_samples()
        y_test_pred = model.get_test_labels()

        if y_train_pred is not None and len(y_train_pred.shape) == 1:
            self.nb_classes = self.get_nb_classes(y_train_pred)
            y_train_pred = check_and_transform_label_format(y_train_pred, nb_classes=self.nb_classes)
        if y_test_pred is not None and len(y_test_pred.shape) == 1:
            if self.nb_classes is None:
                self.nb_classes = self.get_nb_classes(y_test_pred)
            y_test_pred = check_and_transform_label_format(y_test_pred, nb_classes=self.nb_classes)

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

        self.nb_classes = self.get_nb_classes(y_pred)
        predict_fn = (x_pred, y_pred)
        self._art_model = BlackBoxClassifier(predict_fn, x_pred.shape[1:], self.nb_classes, fuzzy_float_compare=True)

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
        return self._art_model.predict(x.get_samples())

    def score(self, test_data: Dataset, scoring_method: Optional[ScoringMethod] = ScoringMethod.ACCURACY, **kwargs):
        """
        Score the model using test data.

        :param test_data: Test data.
        :type train_data: `Dataset`
        :param scoring_method: The method for scoring predictions. Default is ACCURACY.
        :type scoring_method: `ScoringMethod`, optional
        :return: the score as float (for classifiers, between 0 and 1)
        """
        predicted = self._art_model.predict(test_data.get_samples())
        y = check_and_transform_label_format(test_data.get_labels(), nb_classes=self.nb_classes)
        if scoring_method == ScoringMethod.ACCURACY:
            return np.count_nonzero(np.argmax(y, axis=1) == np.argmax(predicted, axis=1)) / predicted.shape[0]
        else:
            raise NotImplementedError
