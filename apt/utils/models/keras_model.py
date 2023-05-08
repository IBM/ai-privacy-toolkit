from typing import Optional

import numpy as np

from sklearn.metrics import mean_squared_error

from apt.utils.models import Model, ModelOutputType, ScoringMethod, check_correct_model_output
from apt.utils.datasets import Dataset, OUTPUT_DATA_ARRAY_TYPE

from art.utils import check_and_transform_label_format
from art.estimators.classification.keras import KerasClassifier as ArtKerasClassifier
from art.estimators.regression.keras import KerasRegressor as ArtKerasRegressor


class KerasModel(Model):
    """
    Wrapper class for keras models.
    """


class KerasClassifier(KerasModel):
    """
    Wrapper class for keras classification models.

    :param model: The original keras model object.
    :type model: `keras.models.Model`
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
    """
    def __init__(self, model: "keras.models.Model", output_type: ModelOutputType, black_box_access: Optional[bool] = True,
                 unlimited_queries: Optional[bool] = True, **kwargs):
        super().__init__(model, output_type, black_box_access, unlimited_queries, **kwargs)
        logits = False
        if output_type == ModelOutputType.CLASSIFIER_LOGITS:
            logits = True
        self._art_model = ArtKerasClassifier(model, use_logits=logits)

    def fit(self, train_data: Dataset, **kwargs) -> None:
        """
        Fit the model using the training data.

        :param train_data: Training data. Labels are expected to either be one-hot encoded or a 1D-array of categorical
                           labels (consecutive integers starting at 0).
        :type train_data: `Dataset`
        :return: None
        """
        y_encoded = check_and_transform_label_format(train_data.get_labels(), self._art_model.nb_classes)
        self._art_model.fit(train_data.get_samples(), y_encoded, **kwargs)

    def predict(self, x: Dataset, **kwargs) -> OUTPUT_DATA_ARRAY_TYPE:
        """
        Perform predictions using the model for input `x`.

        :param x: Input samples.
        :type x: `Dataset`
        :return: Predictions from the model as numpy array (class probabilities, if supported).
        """
        predictions = self._art_model.predict(x.get_samples(), **kwargs)
        check_correct_model_output(predictions, self.output_type)
        return predictions

    def score(self, test_data: Dataset, scoring_method: Optional[ScoringMethod] = ScoringMethod.ACCURACY, **kwargs):
        """
        Score the model using test data.

        :param test_data: Test data.
        :type train_data: `Dataset`
        :param scoring_method: The method for scoring predictions. Default is ACCURACY.
        :type scoring_method: `ScoringMethod`, optional
        :return: the score as float (between 0 and 1)
        """
        y = check_and_transform_label_format(test_data.get_labels(), self._art_model.nb_classes)
        predicted = self.predict(test_data)
        if scoring_method == ScoringMethod.ACCURACY:
            return np.count_nonzero(np.argmax(y, axis=1) == np.argmax(predicted, axis=1)) / predicted.shape[0]
        else:
            raise NotImplementedError


class KerasRegressor(KerasModel):
    """
    Wrapper class for keras regression models.

    :param model: The original keras model object.
    :type model: `keras.models.Model`
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
    def __init__(self, model: "keras.models.Model", black_box_access: Optional[bool] = True,
                 unlimited_queries: Optional[bool] = True, **kwargs):
        super().__init__(model, ModelOutputType.REGRESSOR_SCALAR, black_box_access, unlimited_queries, **kwargs)
        self._art_model = ArtKerasRegressor(model)

    def fit(self, train_data: Dataset, **kwargs) -> None:
        """
        Fit the model using the training data.

        :param train_data: Training data.
        :type train_data: `Dataset`
        :return: None
        """
        self._art_model.fit(train_data.get_samples(), train_data.get_labels(), **kwargs)

    def predict(self, x: Dataset, **kwargs) -> OUTPUT_DATA_ARRAY_TYPE:
        """
        Perform predictions using the model for input `x`.

        :param x: Input samples.
        :type x: `Dataset`
        :return: Predictions from the model as numpy array.
        """
        return self._art_model.predict(x.get_samples(), **kwargs)

    def score(self, test_data: Dataset, scoring_method: Optional[ScoringMethod] = ScoringMethod.MEAN_SQUARED_ERROR,
              **kwargs):
        """
        Score the model using test data.

        :param test_data: Test data.
        :type train_data: `Dataset`
        :param scoring_method: The method for scoring predictions. Default is ACCURACY.
        :type scoring_method: `ScoringMethod`, optional
        :return: the score as float
        """
        predicted = self.predict(test_data)
        if scoring_method == ScoringMethod.MEAN_SQUARED_ERROR:
            return mean_squared_error(test_data.get_labels(), predicted)
        else:
            raise NotImplementedError('Only MEAN_SQUARED_ERROR supported as scoring method')
