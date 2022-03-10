import numpy as np
from sklearn.preprocessing import OneHotEncoder

from apt.utils.models import Model, ModelWithLoss, SingleOutputModel

from art.estimators.classification.scikitlearn import SklearnClassifier as ArtSklearnClassifier
from art.estimators.regression.scikitlearn import ScikitlearnRegressor


class SklearnModel(Model):
    """
    Wrapper class for scikitlearn models.
    """
    def score(self, x: np.ndarray, y: np.ndarray, **kwargs):
        """
        Score the model using test data `(x, y)`.

        :param x: Test data.
        :type x: `np.ndarray` or `pandas.DataFrame`
        :param y: True labels.
        :type y: `np.ndarray` or `pandas.DataFrame`
        """
        return self.model.score(x, y, **kwargs)


class SklearnClassifier(SklearnModel):
    """
    Wrapper class for scikitlearn classification models.
    """
    def __init__(self, model, **kwargs):
        """
        Initialize a `SklearnClassifier` wrapper object.

        :param model: The original sklearn model object
        """
        super().__init__(model, **kwargs)
        self._art_model = ArtSklearnClassifier(model)

    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Fit the model using the training data `(x, y)`.

        :param x: Training data.
        :type x: `np.ndarray` or `pandas.DataFrame`
        :param y: True labels.
        :type y: `np.ndarray` or `pandas.DataFrame`
        """
        encoder = OneHotEncoder(sparse=False)
        if type(y) == np.ndarray:
            y_encoded = encoder.fit_transform(y.reshape(-1, 1))
        else:
            y_encoded = encoder.fit_transform(y.values.reshape(-1, 1))
        self._art_model.fit(x, y_encoded, **kwargs)

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform predictions using the model for input `x`.

        :param x: Input samples.
        :type x: `np.ndarray` or `pandas.DataFrame`
        :return: Predictions from the model.
        """
        return self._art_model.predict(x, **kwargs)


class SklearnRegressor(SklearnModel, SingleOutputModel, ModelWithLoss):
    """
    Wrapper class for scikitlearn regression models.
    """
    def __init__(self, model, **kwargs):
        """
        Initialize a `SklearnRegressor` wrapper object.

        :param model: The original sklearn model object
        """
        super().__init__(model, **kwargs)
        self._art_model = ScikitlearnRegressor(model)

    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Fit the model using the training data `(x, y)`.

        :param x: Training data.
        :type x: `np.ndarray` or `pandas.DataFrame`
        :param y: True labels.
        :type y: `np.ndarray` or `pandas.DataFrame`
        """
        self._art_model.fit(x, y, **kwargs)

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform predictions using the model for input `x`.

        :param x: Input samples.
        :type x: `np.ndarray` or `pandas.DataFrame`
        :return: Predictions from the model.
        """
        return self._art_model.predict(x, **kwargs)

    def loss(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute the loss of the model for samples `x`.

        :param x: Input samples.
        :type x: `np.ndarray` or `pandas.DataFrame`
        :param y: True labels.
        :type y: `np.ndarray` or `pandas.DataFrame`
        :return: Loss values.
        """
        return self._art_model.compute_loss(x, y, **kwargs)


# Probably not needed for now, as we will not be using these wrappers directly in ART.
# class SklearnDecisionTreeClassifier(SklearnClassifier, MultipleOutputModel):
#     """
#     Wrapper class for scikitlearn decision tree classifier models.
#     """
#     def __init__(self, model):
#         """
#         Initialize a `DecisionTreeClassifier` wrapper object.
#
#         :param model: The original sklearn decision tree model object
#         """
#         super().__init__(model)
#         self._art_model = ScikitlearnDecisionTreeClassifier(model)
#
#     def get_decision_path(self, x: np.ndarray) -> np.ndarray:
#         """
#         Returns the nodes along the path taken in the tree when classifying x. Last node is the leaf, first node is the
#         root node.
#
#         :param x: Input samples.
#         :type x: `np.ndarray` or `pandas.DataFrame`
#         :return: The indices of the nodes in the array structure of the tree.
#         """
#         return self._art_model.get_decision_path(x)
#
#     def get_samples_at_node(self, node_id: int) -> int:
#         """
#         Returns the number of training samples mapped to a node.
#
#         :param node_id: The ID of the node.
#         :return: Number of samples mapped this node.
#         """
#         return self._art_model.get_samples_at_node(node_id)
