from abc import ABCMeta, abstractmethod
from typing import Any
from enum import Enum, auto

from apt.utils.datasets import Dataset, OUTPUT_DATA_ARRAY_TYPE


class ModelOutputType(Enum):
    CLASSIFIER_VECTOR = auto()  # probabilities or logits
    CLASSIFIER_SCALAR = auto()  # label only
    REGRESSOR_SCALAR = auto()   # value


class Model(metaclass=ABCMeta):
    """
    Abstract base class for ML model wrappers.
    """

    def __init__(self, model: Any, output_type: ModelOutputType, **kwargs):
        """
        Initialize a `Model` wrapper object.

        :param model: The original model object (of the underlying ML framework)
        :param output_type: The type of output the model yields (vector/label only for classifiers,
                            value for regressors)
        """
        self._model = model
        self._output_type = output_type

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
