from itertools import accumulate

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, MetaEstimatorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from typing import Union
import numpy as np
from sklearn.tree._tree import Tree
from aix360.algorithms.shap import KernelExplainer


# from AIX360.aix360.algorithms.shap import KernelExplainer

class Minimizer():  # BaseEstimator, MetaEstimatorMixin, TransformerMixin):
    # TODO: add type hints and fix when integrating, check if need cells
    def __init__(self, estimator, data_encoder=None,
                 target_accuracy: float = 0.998, categorical_features: Union[np.ndarray, list] = None,
                 features_to_minimize: Union[np.ndarray, list] = None, train_only_QI: bool = True,
                 ):
        self._estimator = estimator
        self._data_encoder = data_encoder
        self._target_accuracy = target_accuracy
        self._categorical_features = categorical_features
        self._features_to_minimize = features_to_minimize
        self._train_only_QI = train_only_QI
        self._feature_dts: Union[dict, None] = None

    def _calc_numerical_dt(self):
        """ Calculates a per feature decision tree for numerical features

        :return:
        """

    def _calc_categorical_dt(self):
        """ Calculates a per feature decision tree for categorical features

        :return:
        """

    @staticmethod
    def get_feature_indices(numerical_features, categorical_features, categorical_encoder):
        numerical_indices = {feature: [i] for i, feature in enumerate(numerical_features)}
        base = len(numerical_indices)
        categorical_indices = {}
        for feature, categories in zip(categorical_features, categorical_encoder.categories_):
            categorical_indices[feature] = list(range(base, base + len(categories)))
            base = base + len(categories)

        return {**numerical_indices, **categorical_indices}
    @classmethod
    def _calc_dt_splits(cls, dt: Tree, node_id: int):
        if node_id == -1:
            return 0
        left_id = dt.children_left[node_id]
        right_id = dt.children_right[node_id]
        return (1 if ((left_id != -1) or (right_id != -1)) else 0) + \
               cls._calc_dt_splits(dt, right_id) + cls._calc_dt_splits(dt, left_id)

    def fit(self, X: pd.DataFrame, y=None):
        numeric_transformer = Pipeline(
            steps=[('imputer', SimpleImputer(strategy='constant', fill_value=0))]
        )
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)
        self._features_to_minimize = self._features_to_minimize if self._features_to_minimize is not None else \
            X.columns
        numeric_features = [f for f in self._features_to_minimize if f not in self._categorical_features and
                            f in self._features_to_minimize]

        self._data_encoder = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, self._categorical_features),
            ]
        )
        # ########
        # categorical_transformer.fit(X[self._categorical_features])
        # ########
        X = self._data_encoder.fit_transform(X)
        columns = list(numeric_features) + list(self._data_encoder.named_transformers_["cat"].get_feature_names())
        # X[]
        X = pd.DataFrame(X, columns=columns)
        y = y if y is not None else self._estimator.predict(X)

        # Calculate decision-tree on all features. get number of splits and set as max depth.
        # node_id=0 means root
        max_depth = self._calc_dt_splits(DecisionTreeClassifier().fit(X, y).tree_, node_id=0)

        feature_categories = {
            feature_name: [name for name in X.columns if name.startswith(f"x{i}_")]
            for i, feature_name in enumerate(self._categorical_features)
        }
        categorical_dts = {
            feature_name: DecisionTreeClassifier(max_depth=max_depth).fit(X[feature_categories[feature_name]], y)
            # something:
            for feature_name in self._categorical_features
        }
        numerical_dts = {
            feature_name: DecisionTreeClassifier(max_depth=max_depth).fit(X[[feature_name]], y)
            for feature_name in numeric_features
        }
        self._feature_dts = {**numerical_dts, **categorical_dts}

        explainer = KernelExplainer(self._estimator.predict, X)
        shap_values = explainer.explain_instance(X)
        global_shap_like_encoded = np.sum(sum(abs(shap_matrix) for shap_matrix in shap_values), axis=0)
        global_shap_categorical_like_encoded = global_shap_like_encoded[self._data_encoder.output_indices_["cat"]]

        # Are the indices here given correctly? Do we need to get them from column transformer?
        global_shap_categorical = {
            # feature: np.sum(global_shap_OHE_like[indices])
            feature: np.sum(global_shap_categorical_like_encoded[indices])
            for feature, indices
            in zip(self._categorical_features, categorical_transformer.categories_)
        }
        global_shap_numerical = {
            feature: shap
            for feature, shap
            in zip(numeric_features, global_shap_like_encoded[self._data_encoder.output_indices_["num"]])
        }

        global_shap = {**global_shap_numerical, **global_shap_categorical}
        # Order features_dts according to heuristic (here it is SHAP)
        sorted_global_shap = sorted(zip(global_shap.values(), self._feature_dts, global_shap.keys()))

        for shap_value, feature_dt, feature in sorted_global_shap:
            pass

        # Prune features_dts

    def transform(self, X):
        pass
