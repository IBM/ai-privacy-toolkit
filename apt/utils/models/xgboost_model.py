from typing import Optional, Tuple

from apt.utils.models import Model, ModelOutputType, ScoringMethod, check_correct_model_output, is_one_hot
from apt.utils.datasets import Dataset, OUTPUT_DATA_ARRAY_TYPE

import numpy as np

from art.estimators.classification.xgboost import XGBoostClassifier as ArtXGBoostClassifier


class XGBoostModel(Model):
    """
    Wrapper class for xgboost models.
    """


class XGBoostClassifier(XGBoostModel):
    """
    Wrapper class for xgboost classification models.

    :param model: The original xgboost model object. Must be fit.
    :type model: Booster or XGBClassifier object
    :param output_type: The type of output the model yields (vector/label only)
    :type output_type: `ModelOutputType`
    :param input_shape: Shape of input to the model.
    :type input_shape: Tuple[int, ...]
    :param nb_classes: Number of prediction classes of the model.
    :type  nb_classes: int
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
    def __init__(self, model: "xgboost.XGBClassifier", output_type: ModelOutputType, input_shape: Tuple[int, ...],
                 nb_classes: int, black_box_access: Optional[bool] = True,
                 unlimited_queries: Optional[bool] = True, **kwargs):
        super().__init__(model, output_type, black_box_access, unlimited_queries, **kwargs)
        self._art_model = ArtXGBoostClassifier(model, nb_features=input_shape[0], nb_classes=nb_classes)
        self.nb_classes = nb_classes

    def fit(self, train_data: Dataset, **kwargs) -> None:
        """
        Fit the model using the training data.

        :param train_data: Training data. Labels are expected to either be one-hot encoded or a 1D-array of categorical
                           labels (consecutive integers starting at 0).
        :type train_data: `Dataset`
        :return: None
        """
        self._art_model._model.fit(train_data.get_samples(), train_data.get_labels())

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
        :return: the score as float (for classifiers, between 0 and 1)
        """
        y = test_data.get_labels()
        predicted = self.predict(test_data)
        if is_one_hot(predicted):
            predicted = np.argmax(predicted, axis=1)
        if is_one_hot(y):
            y = np.argmax(y, axis=1)
        if scoring_method == ScoringMethod.ACCURACY:
            return np.count_nonzero(y == predicted) / predicted.shape[0]
        else:
            raise NotImplementedError
