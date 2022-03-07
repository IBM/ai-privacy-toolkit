from abc import ABCMeta, abstractmethod
from typing import Any

from apt.utils.datasets import Dataset, DATA_ARRAY_TYPE


class Model(metaclass=ABCMeta):
    """
    Abstract base class for ML model wrappers.
    """

    def __init__(self, model: Any, **kwargs):
        """
        Initialize a `Model` wrapper object.

        :param model: The original model object (of the underlying ML framework)
        """
        self._model = model

    @abstractmethod
    def fit(self, train_data: Dataset, **kwargs) -> None:
        """
        Fit the model using the training data.

        :param train_data: Training data.
        :type train_data: `Dataset`
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: DATA_ARRAY_TYPE, **kwargs) -> DATA_ARRAY_TYPE:
        """
        Perform predictions using the model for input `x`.

        :param x: Input samples.
        :type x: `np.ndarray` or `pandas.DataFrame`
        :return: Predictions from the model.
        """
        raise NotImplementedError

    @property
    def model(self):
        """
        Return the model.

        :return: The model.
        """
        return self._model
