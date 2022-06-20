from itertools import accumulate

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, MetaEstimatorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from typing import Union, Dict, List
import numpy as np
from sklearn.tree._tree import Tree
from aix360.algorithms.shap import KernelExplainer
from sklearn.utils import resample


# from AIX360.aix360.algorithms.shap import KernelExplainer

class Minimizer():  # BaseEstimator, MetaEstimatorMixin, TransformerMixin):
    # TODO: add type hints and fix when integrating, check if need cells
    def __init__(self, estimator, data_encoder=None,
                 target_accuracy: float = 0.998, categorical_features: Union[np.ndarray, list] = None,
                 features_to_minimize: Union[np.ndarray, list] = None, train_only_QI: bool = True, random_state=None
                 ):
        """

        :param estimator: fitted estimator model to be used for minimization
        :param data_encoder: fitted encoder that transforms data to match model expectation. Currently, assumes encoder
        has 2 transformers: "num" for numerical and "cat" with respective order.
        :type ColumnTransformer
        :param target_accuracy:
        :param categorical_features: categorical feature names. In the order expected by the encoder.
        :param features_to_minimize: Currently not implemented. Please do not use.
        :param train_only_QI: Currently not implemented. Please do not use.
        :param random_state:
        """
        self._random_state = random_state
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
    def _get_feature_indices(numerical_features, categorical_features, categorical_encoder):
        """

        :param numerical_features: numerical feature names in the order the estimator/encoder expects.
        :param categorical_features:  categorical feature names in the order the estimator/encoder expects.
        :param categorical_encoder: fitted OHE used to encode categorical data.
        :type OneHotEncoder
        :return: a dict of indices lists. Keys are all feature names from output. A value is a list of indices to
        corresponding feature name.
        :type Dict[str, List[int]]
        """
        numerical_indices = {feature: [i] for i, feature in enumerate(numerical_features)}
        base = len(numerical_indices)
        categorical_indices = {}
        for feature, categories in zip(categorical_features, categorical_encoder.categories_):
            categorical_indices[feature] = list(range(base, base + len(categories)))
            base = base + len(categories)

        return {**numerical_indices, **categorical_indices}

    @classmethod
    def _calc_dt_splits(cls, dt: Tree, root_id: int):
        """
        Calculates the number of splits of a subtree given a decision tree and a node id.
        :param dt: sklearn tree object that describes the structure of the decision tree
        :type Tree
        :param root_id: subtree root node id
        :return: number of splits of subtree
        :type int
        """
        if root_id == -1:
            return 0
        left_id = dt.children_left[root_id]
        right_id = dt.children_right[root_id]
        return (1 if ((left_id != -1) or (right_id != -1)) else 0) + cls._calc_dt_splits(dt, right_id) + \
               cls._calc_dt_splits(dt, left_id)

    @staticmethod
    def _init_encoder(numerical_features, categorical_features):
        """
        Initializes a new encoder that supports numercial and categorical features.
        :param numerical_features: numerical features in the order the estimator model expects
        :param categorical_features: categorical features in the order the estimator model expects
        :return: unfitted encoder that appends OHE to numerical features. Sub transformers are named "num"
         for numeric transformer and "cat" for categorical transformer.
        """
        numeric_transformer = Pipeline(
            steps=[('imputer', SimpleImputer(strategy='constant', fill_value=0))]
        )
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)
        return ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numerical_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

    @staticmethod
    def _get_generalizations_from_dts(dts: Dict[str, Tree], numerical_features, categorical_features):
        # TODO: Implement
        raise NotImplementedError
        numerical_thresholds = {
            # TODO: MAKE UNIQUE
            feature_name: dts[feature_name].threshold
            for feature_name in numerical_features
        }
        categories_per_feature = {
            # ...
        }
        untouched_features = [
            # ...
        ]
        return {
            "ranges": numerical_thresholds,
            "categories": categories_per_feature,
            "untouched": untouched_features,
        }

    @staticmethod
    def _calculate_global_shap(estimator, X: pd.DataFrame, background_size, feature_indices: Dict[str, List[int]],
                               random_state):
        """
        Calculates global shap per feature.
        :param estimator:
        :param X:
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
        shap_values = explainer.explain_instance(X)
        global_shap_like_encoded = np.sum(sum(abs(shap_matrix) for shap_matrix in shap_values), axis=0)
        return {
            feature_name: np.sum(global_shap_like_encoded[indices])
            for feature_name, indices in feature_indices.items()
        }

    def fit(self, X: pd.DataFrame, y=None):
        # Get features to minimize form X if non are specified in __init__
        self._features_to_minimize = self._features_to_minimize if self._features_to_minimize is not None else \
            X.columns
        # All non categorical features are numerical
        categorical_features = self._categorical_features
        numerical_features = [f for f in self._features_to_minimize if f not in categorical_features and
                            f in self._features_to_minimize]
        all_features = numerical_features + categorical_features

        # Initialize and fit new encoder if non was supplied by the user.
        if self._data_encoder is None:
            self._data_encoder = self._init_encoder(numerical_features, categorical_features)
            self._data_encoder.fit(X)

        # Encode X and make into pd.Dataframe.
        encoded_columns_names = \
            numerical_features + list(self._data_encoder.named_transformers_["cat"].get_feature_names())
        X = pd.DataFrame(self._data_encoder.transform(X), columns=encoded_columns_names)

        # Calculate predictions of the estimator on unencoded data if not supplied by user.
        if y is None:
            y = self._estimator.predict(X)

        # Train decision-tree on all features. get number of splits and set as max depth.
        # root_id=0 means root
        max_depth = self._calc_dt_splits(DecisionTreeClassifier().fit(X, y).tree_, root_id=0)

        feature_indices = self._get_feature_indices(numerical_features, categorical_features,
                                                    self._data_encoder.named_transformers_["cat"])

        self._feature_dts = {
            feature_name: DecisionTreeClassifier(max_depth=max_depth).fit(X.iloc[:, indices], y)
            for feature_name, indices in feature_indices.items()
        }

        # global_shap_dict =\
        #     self._calculate_global_shap(estimator=self._estimator, X=X, background_size=100,
        #                                 feature_indices=feature_indices, random_state=self._random_state)
        global_shap_dict = {feature_name: i for i, feature_name in enumerate(all_features)}

        # Order features_dts according to heuristic (here it is SHAP)
        shap_sorted_features = sorted(list(global_shap_dict.items()), key=lambda tup: tup[1])
        for feature_name, shap_value in shap_sorted_features:
            # TODO: Implement pruning and use here.
            dt = self._feature_dts[feature_name]

    def transform(self, X):
        raise NotImplementedError
