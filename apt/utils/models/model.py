from abc import ABCMeta, abstractmethod
from typing import Any, Optional
from enum import Enum, auto

from apt.utils.datasets import Dataset, OUTPUT_DATA_ARRAY_TYPE


class ModelOutputType(Enum):
    CLASSIFIER_VECTOR = auto()  # probabilities or logits
    CLASSIFIER_SCALAR = auto()  # label only
    REGRESSOR_SCALAR = auto()  # value


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
        :type x: `np.ndarray` or `pandas.DataFrame`
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
