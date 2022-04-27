from typing import Optional

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
    def __init__(self, model: BaseEstimator, output_type: ModelOutputType, black_box_access: Optional[bool] = True,
                 unlimited_queries: Optional[bool] = True, **kwargs):
        """
        Initialize a `SklearnClassifier` wrapper object.

        :param model: The original sklearn model object.
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
        super().__init__(model, output_type, black_box_access, unlimited_queries, **kwargs)
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
    def __init__(self, model: BaseEstimator, black_box_access: Optional[bool] = True,
                 unlimited_queries: Optional[bool] = True, **kwargs):
        """
        Initialize a `SklearnRegressor` wrapper object.

        :param model: The original sklearn model object.
        :param black_box_access: Boolean describing the type of deployment of the model (when in production).
                                 Set to True if the model is only available via query (API) access, i.e.,
                                 only the outputs of the model are exposed, and False if the model internals
                                 are also available. Optional, Default is True.
        :param unlimited_queries: If black_box_access is True, this boolean indicates whether a user can perform
                                  unlimited queries to the model API or whether there is a limit to the number of
                                  queries that can be submitted. Optional, Default is True.
        """
        super().__init__(model, ModelOutputType.REGRESSOR_SCALAR, black_box_access, unlimited_queries, **kwargs)
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
