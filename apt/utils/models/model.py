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
    """

    def __init__(self, model: Any, output_type: ModelOutputType, black_box_access: Optional[bool] = True,
                 unlimited_queries: Optional[bool] = True, **kwargs):
        """
        Initialize a `Model` wrapper object.

        :param model: The original model object (of the underlying ML framework)
        :param output_type: The type of output the model yields (vector/label only for classifiers,
                            value for regressors)
        :param black_box_access: Boolean describing the type of deployment of the model (when in production).
                                 Set to True if the model is only available via query (API) access, i.e.,
                                 only the outputs of the model are exposed, and False if the model internals
                                 are also available. Optional, Default is True.
        :param unlimited_queries: If black_box_access is True, this boolean indicates whether a user can perform
                                  unlimited queries to the model API or whether there is a limit to the number of
                                  queries that can be submitted. Optional, Default is True.
        """
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
        :return: Predictions from the model.
        """
        raise NotImplementedError

    @abstractmethod
    def score(self, test_data: Dataset, **kwargs):
        """
        Score the model using test data.

        :param test_data: Test data.
        :type train_data: `Dataset`
        """
        return NotImplementedError

    @property
    def model(self) -> Any:
        """
        Return the model.

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
    def black_box_access(self) -> Any:
        """
        Return True if the model is only available via query (API) access, i.e.,
        only the outputs of the model are exposed, and False if the model internals are also available.

        :return: True if the model is only available via query (API) access, i.e.,
                 only the outputs of the model are exposed, and False if the model internals are also available.
        """
        return self._black_box_access

    @property
    def unlimited_queries(self) -> Any:
        """
        If black_box_access is True, Return whether a user can perform unlimited queries to the model API
        or whether there is a limit to the number of queries that can be submitted.

        :return: If black_box_access is True, Return whether a user can perform unlimited queries to the model API
                 or whether there is a limit to the number of queries that can be submitted.
        """
        return self._unlimited_queries
