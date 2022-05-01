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
        :return the score as float (for classifiers, between 0 and 1)
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
        :type model: scikitlearn classifier object
        :param output_type: The type of output the model yields (vector/label only)
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
        :return None
        """
        super().__init__(model, output_type, black_box_access, unlimited_queries, **kwargs)
        self._art_model = ArtSklearnClassifier(model)

    def fit(self, train_data: Dataset, **kwargs) -> None:
        """
        Fit the model using the training data.

        :param train_data: Training data.
        :type train_data: `Dataset`
        :return None
        """
        encoder = OneHotEncoder(sparse=False)
        y_encoded = encoder.fit_transform(train_data.get_labels().reshape(-1, 1))
        self._art_model.fit(train_data.get_samples(), y_encoded, **kwargs)

    def predict(self, x: Dataset, **kwargs) -> OUTPUT_DATA_ARRAY_TYPE:
        """
        Perform predictions using the model for input `x`.

        :param x: Input samples.
        :type x: `Dataset`
        :return: Predictions from the model as numpy array (class probabilities, if supported).
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
        :type model: scikitlearn regressor object
        :param black_box_access: Boolean describing the type of deployment of the model (when in production).
                                 Set to True if the model is only available via query (API) access, i.e.,
                                 only the outputs of the model are exposed, and False if the model internals
                                 are also available. Default is True.
        :type black_box_access: boolean, optional
        :param unlimited_queries: If black_box_access is True, this boolean indicates whether a user can perform
                                  unlimited queries to the model API or whether there is a limit to the number of
                                  queries that can be submitted. Default is True.
        :type unlimited_queries: boolean, optional
        :return None
        """
        super().__init__(model, ModelOutputType.REGRESSOR_SCALAR, black_box_access, unlimited_queries, **kwargs)
        self._art_model = ScikitlearnRegressor(model)

    def fit(self, train_data: Dataset, **kwargs) -> None:
        """
        Fit the model using the training data.

        :param train_data: Training data.
        :type train_data: `Dataset`
        :return None
        """
        self._art_model.fit(train_data.get_samples(), train_data.get_labels(), **kwargs)

    def predict(self, x: Dataset, **kwargs) -> OUTPUT_DATA_ARRAY_TYPE:
        """
        Perform predictions using the model for input `x`.

        :param x: Input samples.
        :type x: `Dataset`
        :return: Predictions from the model as numpy array.
        """
        return self._art_model.predict(x, **kwargs)
