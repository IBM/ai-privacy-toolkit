import pandas as pd
import numpy as np
from typing import List, Dict, Union
from . import OrderedFeatureMinimizer
from aix360.algorithms.shap import KernelExplainer
from sklearn.utils import resample


class ShapMinimizer(OrderedFeatureMinimizer):
    def __init__(self, *args, n_samples, background_size, **kwargs):
        self._n_samples = n_samples
        self._background_size = background_size
        super().__init__(*args, **kwargs, ordered_features=None)

    def _get_ordered_features(self, estimator, encoder, X_train, y_train, numerical_features, categorical_features,
                              feature_indices, random_state=None):
        global_shap_dict = \
            self.calculate_global_shap(estimator=estimator, X=X_train, y=y_train,
                                       background_size=self._background_size,
                                       feature_indices=feature_indices, random_state=random_state,
                                       nsamples=self._n_samples)
        shap_sorted_features = sorted(list(global_shap_dict.items()), key=lambda tup: tup[1])
        self._ordered_shap_features = [feature for feature, _ in shap_sorted_features]
        return self._ordered_shap_features

    @property
    def ordered_shap_features(self):
        return self._ordered_shap_features

    @staticmethod
    def calculate_global_shap(estimator, X: pd.DataFrame, y: Union[pd.DataFrame, np.ndarray],
                              background_size, feature_indices: Dict[str, List[int]], random_state, nsamples):
        """
        Calculates global shap per feature.
        :param nsamples:
        :param estimator:
        :param X:
        :param y:
        :param background_size:
        :param feature_indices: dict
        :type Dict[str, List[int]]
        :param random_state:
        :return: dict of global shap like
        :type Dict[str, float]
        """
        # TODO: Check kmeans instead of resample.Change that if needed.
        background_X = resample(X, n_samples=background_size, random_state=random_state)
        explainer = KernelExplainer(estimator.predict_proba, background_X)
        # The following line picks only the shap values for the correct label for each record.
        shap_values = \
            np.array(explainer.explain_instance(X, nsamples=nsamples)).transpose(1, 0, 2)[np.arange(0, X.shape[0]), y]
        return {
            feature_name: np.sum(abs(shap_values[indices]))
            for feature_name, indices in feature_indices.items()
        }
