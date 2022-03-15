import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator

from apt.utils.models import Model, ModelOutputType
from apt.utils.datasets import Dataset, OUTPUT_DATA_ARRAY_TYPE

from art.estimators.classification.scikitlearn import SklearnClassifier as ArtSklearnClassifier
from art.estimators.regression.scikitlearn import ScikitlearnRegressor


class SklearnModel(Model):
    """
    Wrapper class for scikitlearn models.
    """
    def score(self, test_data: Dataset, **kwargs):
        """
        Score the model using test data.

        :param test_data: Test data.
        :type train_data: `Dataset`
        """
        return self.model.score(test_data.get_samples(), test_data.get_labels(), **kwargs)


class SklearnClassifier(SklearnModel):
    """
    Wrapper class for scikitlearn classification models.
    """
    def __init__(self, model: BaseEstimator, output_type: ModelOutputType, **kwargs):
        """
        Initialize a `SklearnClassifier` wrapper object.

        :param model: The original sklearn model object
        """
        super().__init__(model, output_type, **kwargs)
        self._art_model = ArtSklearnClassifier(model)

    def fit(self, train_data: Dataset, **kwargs) -> None:
        """
        Fit the model using the training data.

        :param train_data: Training data.
        :type train_data: `Dataset`
        """
        encoder = OneHotEncoder(sparse=False)
        y_encoded = encoder.fit_transform(train_data.get_labels().reshape(-1, 1))
        self._art_model.fit(train_data.get_samples(), y_encoded, **kwargs)

    def predict(self, x: Dataset, **kwargs) -> OUTPUT_DATA_ARRAY_TYPE:
        """
        Perform predictions using the model for input `x`.

        :param x: Input samples.
        :type x: `np.ndarray` or `pandas.DataFrame`
        :return: Predictions from the model (class probabilities, if supported).
        """
        return self._art_model.predict(x, **kwargs)


class SklearnRegressor(SklearnModel):
    """
    Wrapper class for scikitlearn regression models.
    """
    def __init__(self, model: BaseEstimator, **kwargs):
        """
        Initialize a `SklearnRegressor` wrapper object.

        :param model: The original sklearn model object
        """
        super().__init__(model, ModelOutputType.REGRESSOR_SCALAR, **kwargs)
        self._art_model = ScikitlearnRegressor(model)

    def fit(self, train_data: Dataset, **kwargs) -> None:
        """
        Fit the model using the training data.

        :param train_data: Training data.
        :type train_data: `Dataset`
        """
        self._art_model.fit(train_data.get_samples(), train_data.get_labels(), **kwargs)

    def predict(self, x: Dataset, **kwargs) -> OUTPUT_DATA_ARRAY_TYPE:
        """
        Perform predictions using the model for input `x`.

        :param x: Input samples.
        :type x: `np.ndarray` or `pandas.DataFrame`
        :return: Predictions from the model.
        """
        return self._art_model.predict(x, **kwargs)
