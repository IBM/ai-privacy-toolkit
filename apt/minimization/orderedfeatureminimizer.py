from copy import deepcopy
from itertools import accumulate

import pandas as pd
import sklearn.metrics
from sklearn.base import BaseEstimator, TransformerMixin, MetaEstimatorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from typing import Union, Dict, List, Set
import numpy as np
from sklearn.tree._tree import Tree
from sklearn.metrics import accuracy_score


class OrderedFeatureMinimizer:  # BaseEstimator, MetaEstimatorMixin, TransformerMixin):
    # TODO: add type hints and fix when integrating, check if need cells
    def __init__(self, estimator, data_encoder=None,
                 target_accuracy: float = 0.998, categorical_features: Union[np.ndarray, list] = None,
                 features_to_minimize: Union[np.ndarray, list] = None, train_only_QI: bool = True, random_state=None,
                 ordered_features: List[str] = None
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
        self._ordered_features = ordered_features
        self._random_state = random_state
        self._estimator = estimator
        self._data_encoder = data_encoder
        self._target_accuracy = target_accuracy
        self._categorical_features = categorical_features
        self._features_to_minimize = features_to_minimize
        self._train_only_QI = train_only_QI
        self._feature_dts: Union[dict, None] = None

        self._numerical_features = None
        self._generalizations = None
        self._categories_dict = None
        self._depths: Dict[str, int]
        self._generalization_arrays: Dict[str, List]
        self._untouched_features: List[str]

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

    @classmethod
    def _get_numerical_generalization(cls, dt: Tree, depth: int, medians, thresholds=None, node_id=0):
        """

        :param dt: single feature decision tree trained on numerical data
        :param depth:
        :return: thresholds of generalization.
        :type List[int]
        """
        # TODO: Make sure the values returned are always unique
        if thresholds is None:
            thresholds = []
        left_id = dt.children_left[node_id]
        right_id = dt.children_right[node_id]
        if (left_id < 0 or right_id < 0) or depth == 0 or medians[left_id] is None or medians[right_id] is None:
            return thresholds
        thresholds.append(dt.threshold[node_id])
        cls._get_numerical_generalization(dt, depth - 1, medians, thresholds, left_id)
        cls._get_numerical_generalization(dt, depth - 1, medians, thresholds, right_id)
        return thresholds

    @classmethod
    def _get_categorical_generalization(cls, dt: Tree, depth: int, majorities, node_id=0, categories=None):
        """
        :param categories: categories of feature in the order of OHE.
        :param dt: single feature decision tree trained on categorical OHE data.
        :return: set that represents a generalized category (a dt can only have one generalized category)
        """
        # dt.features contains which category each node is split by or a negative if none.
        if categories is None:
            categories = set(range(dt.n_features))
        left_id = dt.children_left[node_id]
        right_id = dt.children_right[node_id]
        if (left_id < 0 or right_id < 0) or depth == 0 or majorities[left_id] is None or majorities[right_id] is None:
            categories.discard(majorities[node_id])
            return categories
        cls._get_categorical_generalization(dt, depth - 1, majorities, left_id, categories)
        cls._get_categorical_generalization(dt, depth - 1, majorities, right_id, categories)
        return categories

    @classmethod
    def _get_generalizations_from_dts(cls, dts: Dict[str, DecisionTreeClassifier], numerical_features: List[str],
                                      categorical_features: List[str], untouched_features: List[str],
                                      categories_dict: Dict[str, List[str]],
                                      generalization_arrays: Dict[str, List], depths: Dict[str, int]):
        """

        :param dts: feature decision trees
        :param numerical_features: numerical feature names
        :param categorical_features: categorical feature names
        :param categories_dict: categories of each feature by feature name
        :type Dict[str, List[str]]
        :return: thresholds for numerical features, generalized categories for categorical features and a list of
         features that were not generalized at all.
        :type Dict[str, Union[Dict, List]]
        """
        # numerical_thresholds = {}
        # for feature_name in numerical_features:
        #     numerical_thresholds[feature_name] =
        numerical_thresholds = {
            feature_name: cls._get_numerical_generalization(dts[feature_name].tree_, depths[feature_name],
                                              generalization_arrays[feature_name])
            for feature_name in numerical_features
        }

        categories_per_feature = {}
        for feature_name in categorical_features:
            categories_per_feature[feature_name] = [{
                categories_dict[feature_name][category]
                for category in cls._get_categorical_generalization(dts[feature_name].tree_, depths[feature_name],
                                                                    generalization_arrays[feature_name])
            }]

        # categories_per_feature = {
        #     feature_name: [{
        #         categories_dict[category]
        #         for category in cls._get_categorical_generalization(dts[feature_name].tree_, depths[feature_name],
        #                                                             generalization_arrays[feature_name])
        #     }]
        #     for feature_name in categorical_features
        # }

        return {
            "ranges": numerical_thresholds,
            "categories": categories_per_feature,
            "untouched": untouched_features,
        }

    @classmethod
    def _transform_numerical_feature(cls, dt: Tree, X: np.ndarray, medians: list, depth: int, node_id: int = 0):
        if X.size == 0:
            return
        # check if each split results in left and right children
        threshold = dt.threshold[node_id]
        left_id = dt.children_left[node_id]
        right_id = dt.children_right[node_id]
        if (left_id < 0 or right_id < 0) or depth == 0 or medians[left_id] is None or medians[right_id] is None:
            return medians[node_id]
        if left_id >= 0:
            X[X <= threshold] = cls._transform_numerical_feature(dt, X[X <= threshold], medians, depth - 1, left_id)
        if right_id >= 0:
            X[X > threshold] = cls._transform_numerical_feature(dt, X[X > threshold], medians, depth - 1, right_id)
        return X

    @classmethod
    def _transform_categorical_feature(cls, dt: Tree, X: np.ndarray, majors: list, depth: int, node_id: int = 0):
        # TODO Debug and make sure it works
        if X.size == 0:
            return
        split_feature = dt.feature[node_id]
        threshold = dt.threshold[node_id]
        left_id = dt.children_left[node_id]
        right_id = dt.children_right[node_id]
        if (left_id < 0 or right_id < 0) or depth == 0 or majors[left_id] is None or majors[right_id] is None:
            X[:] = 0
            X[:, majors[node_id]] = 1
            return X
        y = X[:, split_feature]
        if left_id >= 0:
            X[y <= threshold] = cls._transform_categorical_feature(dt, X[y <= threshold], majors, depth - 1,
                                                                   left_id)
        if right_id >= 0:
            X[y > threshold] = cls._transform_categorical_feature(dt, X[y > threshold], majors, depth - 1,
                                                                  right_id)
        return X

    @classmethod
    def _transform_in_place(cls, dts, X, depths: Dict[str, int], numerical_features, categorical_features,
                            feature_indices, generalizations_arrays):
        for feature in numerical_features:
            index = feature_indices[feature][0]
            X[:, index] = cls._transform_numerical_feature(dts[feature].tree_, X[:, index],
                                                           generalizations_arrays[feature], depths[feature], 0)

        for feature in categorical_features:
            indices = feature_indices[feature]
            X[:, indices] = cls._transform_categorical_feature(dts[feature].tree_, X[:, indices],
                                                               generalizations_arrays[feature], depths[feature], 0)

    @classmethod
    def populate_representative_median(cls, dt: Tree, X: np.ndarray, node_id, representative_values):
        if X.size == 0:
            return
        threshold = dt.threshold[node_id]
        representative_values[node_id] = np.median(X)
        left_id = dt.children_left[node_id]
        right_id = dt.children_right[node_id]
        if left_id >= 0:
            cls.populate_representative_median(dt, X[X <= threshold], left_id, representative_values)
        if right_id >= 0:
            cls.populate_representative_median(dt, X[X > threshold], right_id, representative_values)

    @classmethod
    def populate_representative_majority(cls, dt: Tree, X: np.ndarray, node_id, representative_values):
        # since data is one hot encoded this line gets the majority index
        if X.size == 0:
            return
        representative_values[node_id] = X.sum(axis=0).argmax()
        split_feature = dt.feature[node_id]
        threshold = dt.threshold[node_id]
        left_id = dt.children_left[node_id]
        right_id = dt.children_right[node_id]
        # the threshold for encoded data is 0.5, therefore we take the column of the split feature and test threshold
        # on it in order to split the data between the left and right child
        if left_id >= 0:
            cls.populate_representative_majority(dt, X[:, X[split_feature] <= threshold], left_id,
                                                 representative_values)
        if right_id >= 0:
            cls.populate_representative_majority(dt, X[:, X[split_feature] > threshold], right_id,
                                                 representative_values)

    @classmethod
    def _calculate_tree_depth(cls, dt, node_id=0):
        if node_id < 0:
            return 0
        return 1 + max(cls._calculate_tree_depth(dt, dt.tree_.children_left[node_id]),
                       cls._calculate_tree_depth(dt, dt.tree_.children_right[node_id]))

    def fit(self, X: pd.DataFrame, y=None):
        # Get features to minimize form X if non are specified in __init__
        self._features_to_minimize = self._features_to_minimize if self._features_to_minimize is not None else \
            X.columns
        # All non categorical features are numerical
        categorical_features = self._categorical_features
        numerical_features = self._numerical_features = \
            [f for f in self._features_to_minimize if f not in categorical_features and f in self._features_to_minimize]
        all_features = numerical_features + categorical_features

        # Split to train and test sets.
        if y is None:
            X_train, X_test = train_test_split(X, test_size=0.4)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

        # Initialize and fit new encoder if non was supplied by the user.
        if self._data_encoder is None:
            self._data_encoder = self._init_encoder(numerical_features, categorical_features)
            self._data_encoder.fit(X_train)

        # Get categories
        self._categories_dict = \
            dict(zip(categorical_features, self._data_encoder.named_transformers_["cat"].categories_))

        # Encode X and make into pd.Dataframe.
        encoded_columns_names = \
            numerical_features + list(self._data_encoder.named_transformers_["cat"].get_feature_names())
        X_train = pd.DataFrame(self._data_encoder.transform(X_train), columns=encoded_columns_names)
        X_test = pd.DataFrame(self._data_encoder.transform(X_test), columns=encoded_columns_names)

        # Calculate predictions of the estimator on unencoded data if not supplied by user.
        if y is None:
            y_train = self._estimator.predict(X_train)
            y_test = self._estimator.predict(X_test)

        # Train decision-tree on all features. get number of splits and set as max depth.
        # root_id=0 means root
        max_depth = self._calc_dt_splits(DecisionTreeClassifier().fit(X_train, y_train).tree_, root_id=0)

        feature_indices = self._get_feature_indices(numerical_features, categorical_features,
                                                    self._data_encoder.named_transformers_["cat"])

        self._feature_dts = {
            feature_name: DecisionTreeClassifier(max_depth=max_depth).fit(X_train.iloc[:, indices], y_train)
            for feature_name, indices in feature_indices.items()
        }

        # calculate generalization values (medians and majority)
        self._generalization_arrays = generalization_arrays = {
            feature_name: [None] * self._feature_dts[feature_name].tree_.node_count
            for feature_name in all_features
        }
        for feature_name in numerical_features:
            self.populate_representative_median(self._feature_dts[feature_name].tree_,
                                                X_train.iloc[:, feature_indices[feature_name]].to_numpy(), 0,
                                                generalization_arrays[feature_name])
        for feature_name in categorical_features:
            self.populate_representative_majority(self._feature_dts[feature_name].tree_,
                                                  X_train.iloc[:, feature_indices[feature_name]].to_numpy(), 0,
                                                  generalization_arrays[feature_name])

        # Order features_dts according to heuristic
        self._depths = depths = {feature_name: self._calculate_tree_depth(self._feature_dts[feature_name], 0)
                                 for feature_name in all_features}
        X_test_transformed = np.copy(X_test)

        if self._ordered_features is not None:
            ordered_features = self._ordered_features
        else:
            ordered_features = self._get_ordered_features(self._estimator, self._data_encoder, X_train,
                                                          numerical_features, categorical_features, feature_indices,
                                                          self._random_state)
        untouched_features = self._untouched_features = []

        for feature_name in ordered_features:
            # TODO: Implement pruning and use here.
            # self._transform_in_place(self._feature_dts, X_test_transformed, depths, numerical_features,
            #                          categorical_features,
            #                          feature_indices, generalization_arrays)
            y_transformed = self._estimator.predict(X_test_transformed)
            accuracy = accuracy_score(y_test, y_transformed)
            init_depth = depths[feature_name]
            for i in range(depths[feature_name]):
                if accuracy < self._target_accuracy:
                    break
                depths[feature_name] -= 1
                self._transform_in_place(self._feature_dts, X_test_transformed, depths,
                                         [feature_name] if feature_name in numerical_features else [],
                                         [feature_name] if feature_name in categorical_features else [],
                                         feature_indices, generalization_arrays)
                y_transformed = self._estimator.predict(X_test_transformed)
                accuracy = accuracy_score(y_test, y_transformed)

            if accuracy < self._target_accuracy:
                depths[feature_name] += 1
                X_test_transformed[:, feature_indices[feature_name]] = X_test.iloc[:, feature_indices[feature_name]]
                if depths[feature_name] != init_depth:
                    self._transform_in_place(self._feature_dts, X_test_transformed, depths,
                                             [feature_name] if feature_name in numerical_features else [],
                                             [feature_name] if feature_name in categorical_features else [],
                                             feature_indices,
                                             generalization_arrays)
                else:
                    untouched_features.append(feature_name)

    @classmethod
    def _get_ordered_features(cls, estimator, encoder, X_train, numerical_features, categorical_features,
                              feature_indices, random_state=None):
        return numerical_features + categorical_features

    def transform(self, X: pd.DataFrame):
        X_transformed = self._data_encoder.transform(X)
        filtered_numerical_features = [feature for feature in self._numerical_features if
                                       feature not in self._untouched_features]
        filtered_categorical_features = [feature for feature in self._categorical_features if
                                         feature not in self._untouched_features]
        self._transform_in_place(
            self._feature_dts, X_transformed, self._depths, filtered_numerical_features,
            filtered_categorical_features, self._get_feature_indices(self._numerical_features,
                                                                     self._categorical_features,
                                                                     self._data_encoder.named_transformers_["cat"]),
            self._generalization_arrays
        )
        return X_transformed

    @property
    def generalizations(self):
        # TODO: Implement this. Use untouched features.
        filtered_numerical_features = [feature for feature in self._numerical_features if
                                       feature not in self._untouched_features]
        filtered_categorical_features = [feature for feature in self._categorical_features if
                                         feature not in self._untouched_features]
        if self._generalizations is None:
            return self._get_generalizations_from_dts(self._feature_dts, filtered_numerical_features,
                                                      filtered_categorical_features, self._untouched_features,
                                                      self._categories_dict, self._generalization_arrays,
                                                      self._depths)
