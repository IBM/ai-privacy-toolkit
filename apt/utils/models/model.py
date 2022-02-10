from abc import ABC, abstractmethod
from typing import Union, List, Any, Optional
import numpy as np

class Model(ABC):
    """
    Base class for ML model wrappers.
    """

    def __init__(self, model: Any, **kwargs):
        """
            Initialize a `Model` wrapper object.

            :param model: The original model object (of the underlying ML framework)
        """
        self._model = model

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Fit the model using the training data `(x, y)`.

        :param x: Training data.
        :type x: `np.ndarray` or `pandas.DataFrame`
        :param y: True labels.
        :type y: `np.ndarray` or `pandas.DataFrame`
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
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


class SingleOutputModel(Model):
    """
    Wrapper class for ML models whose output is a single value (e.g., classification with label only output, regression).
    """


class MultipleOutputModel(Model):
    """
    Wrapper class for ML models whose output is a vector (e.g., class probabilities or logits).
    """


class ModelWithLoss(Model):
    """
    Wrapper class for ML models that support computing loss values for predictions.
    """

    def __init__(self, model: Any, loss: Optional[Any] = None, **kwargs):
        """
            Initialize a `ModelWithLoss` wrapper object.

            :param model: The original model object (of the underlying ML framework)
            :param loss: The loss function/object of the model (of the underlying ML framework)
        """
        super().__init__(model, **kwargs)
        self._loss = loss


    # Probably not needed for now, as we will not be using these wrappers directly in ART.
    # @abstractmethod
    # def loss(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
    #     """
    #     Compute the loss of the model for samples `x`.
    #
    #     :param x: Input samples.
    #     :type x: `np.ndarray` or `pandas.DataFrame`
    #     :param y: True labels.
    #     :type y: `np.ndarray` or `pandas.DataFrame`
    #     :return: Loss values.
    #     """
    #     raise NotImplementedError


# Probably not needed for now, as we will not be using these wrappers directly in ART.
# class ModelWithGradients(Model):
#     """
#     Wrapper class for ML models that support computing gradients.
#     """
#     @abstractmethod
#     def class_gradient(self, x: np.ndarray, label: Union[int, List[int], None] = None, **kwargs) -> np.ndarray:
#         """
#         Compute per-class derivatives w.r.t. input `x`.
#
#         :param x: Input samples.
#         :type x: `np.ndarray` or `pandas.DataFrame`
#         :param label: Index of a specific class. If provided, the gradient of the specified class
#                      is computed for all samples. Otherwise, gradients for all classes are computed for all samples.
#         :param label: int
#         :return: Gradients of input features w.r.t. each class in the form `(batch_size, nb_classes, input_shape)` when
#                  computing for all classes, or `(batch_size, 1, input_shape)` when `label` is specified.
#         """
#         raise NotImplementedError
