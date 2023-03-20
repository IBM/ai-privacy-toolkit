from . import OrderedFeatureMinimizer
from sklearn.tree import DecisionTreeClassifier
import numpy as np


class DTImportanceMinimizer(OrderedFeatureMinimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, ordered_features=None, **kwargs)
        self._importance_ordered_features = None

    @property
    def importance_ordered_features(self):
        return self._importance_ordered_features

    def _get_ordered_features(self, estimator, encoder, X_train, y_train, numerical_features, categorical_features,
                              feature_indices, random_state=None):
        dt = DecisionTreeClassifier().fit(X_train, y_train)
        feature_importances = {
            feature: np.sum(dt.feature_importances_[indices])
            for feature, indices in feature_indices.items()
        }
        sorted_features = sorted(feature_importances.items(), key=lambda tup: tup[1])
        self._importance_ordered_features = [feature for feature, _ in sorted_features]
        return self.importance_ordered_features
