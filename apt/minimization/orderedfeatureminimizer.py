from copy import deepcopy
from itertools import accumulate

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, MetaEstimatorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score as uniform_accuracy_score
from typing import Union, Dict, List, Set, Callable
import numpy as np
from sklearn.tree._tree import Tree


# TODO: use mixins correctly
class OrderedFeatureMinimizer:  # (BaseEstimator, MetaEstimatorMixin, TransformerMixin):
    def __init__(self, estimator, encoder=None,
                 target_accuracy: float = 0.998, categorical_features: Union[np.ndarray, list] = None,
                 features_to_minimize: Union[np.ndarray, list] = None, train_only_QI: bool = True, random_state=None,
                 ordered_features: List[str] = None, accuracy_score: Callable = uniform_accuracy_score
                 ):
        """

        :param estimator: fitted estimator model to be used for minimization
        :param encoder: fitted encoder that transforms data to match model expectation. Currently, assumes encoder
        has 2 transformers: "num" for numerical and "cat" with respective order.
        :type ColumnTransformer
        :param target_accuracy:
        :param categorical_features: categorical feature names. In the order expected by the encoder.
        :param features_to_minimize: Currently not implemented. Please do not use.
        :param train_only_QI: Currently not implemented. Please do not use.
        :param random_state:
        :param ordered_features:
        """

        self._estimator = estimator
        self._data_encoder = encoder
        self._target_accuracy = target_accuracy
        self._categorical_features = categorical_features
        self._features_to_minimize = features_to_minimize
        self._train_only_QI = train_only_QI
        self._random_state = random_state
        self._ordered_features = ordered_features
        self._accuracy_score = accuracy_score

        self._numerical_features = None
        self._feature_dts: Union[Dict[str, Tree]]
        self._generalizations = None
        self._categories_dict: Dict
        self._depths: Dict[str, int]
        self._generalization_arrays: Dict[str, List]
        self._untouched_features: List[str]
        self.feature_indices = None

    @staticmethod
    def _get_feature_indices(numerical_features, categorical_features, categorical_encoder: OneHotEncoder):
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
    def _get_categorical_generalization(cls, dt: Tree, depth: int, majorities, node_id=0,
                                        categories: Union[None, Set] = None):
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
            return categories
        categories.discard(dt.feature[node_id])
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
        numerical_thresholds = {
            feature_name: cls._get_numerical_generalization(dts[feature_name].tree_, depths[feature_name],
                                                            generalization_arrays[feature_name])
            for feature_name in {feature for feature in numerical_features if feature not in untouched_features}
        }

        categories_per_feature = {}
        untouched_features = untouched_features.copy()
        for feature_name in {feature for feature in categorical_features if feature not in untouched_features}:
            generalization = [{
                categories_dict[feature_name][category]
                for category in cls._get_categorical_generalization(dts[feature_name].tree_, depths[feature_name],
                                                                    generalization_arrays[feature_name])
            }]
            if len(generalization[0]) > 1:
                categories_per_feature[feature_name] = generalization
            else:
                untouched_features.append(feature_name)

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
        # Train set is used to train the feature dts
        # Test set is used for ordering data and for pruning
        if y is None:
            X_train, X_test = train_test_split(X, test_size=0.4, random_state=self._random_state)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=self._random_state)

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

        # Train decision-tree on all features. get number of splits and set as max depth for training feature dts.
        # root_id=0 means root
        max_depth = self._calc_dt_splits(
            DecisionTreeClassifier(random_state=self._random_state).fit(X_train, y_train).tree_, root_id=0)

        self.feature_indices = feature_indices = self._get_feature_indices(numerical_features, categorical_features,
                                                                           self._data_encoder.named_transformers_[
                                                                               "cat"])

        self._feature_dts = {
            feature_name:
                DecisionTreeClassifier(max_depth=max_depth, random_state=self._random_state)
                .fit(X_train.iloc[:, indices], y_train)
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

        self._depths = depths = {feature_name: self._calculate_tree_depth(self._feature_dts[feature_name], 0)
                                 for feature_name in all_features}

        # Order features_dts according to heuristic
        if self._ordered_features is not None:
            ordered_features = self._ordered_features
        else:
            ordered_features = self._get_ordered_features(self._estimator, self._data_encoder, X_train, y_train,
                                                          numerical_features, categorical_features, feature_indices,
                                                          self._random_state)

        # Prune dts based on target accuracy using test set
        # Depths is changed inplace
        self._untouched_features = self._prune(
            estimator=self._estimator,
            feature_dts=self._feature_dts,
            X_test=X_test,
            y_test=y_test,
            ordered_features=ordered_features,
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            feature_indices=feature_indices,
            generalization_arrays=generalization_arrays,
            depths=depths,
            target_accuracy=self._target_accuracy,
            accuracy_score=self._accuracy_score
        )
        return self

    @classmethod
    def _prune(cls, estimator, feature_dts, X_test, y_test, ordered_features, numerical_features, categorical_features,
               feature_indices, generalization_arrays, depths, target_accuracy, accuracy_score):
        untouched_features = []
        X_test_transformed = np.copy(X_test)
        y_transformed = estimator.predict(X_test_transformed)
        for feature_name in ordered_features:
            indices = feature_indices[feature_name]
            numerical_features_to_transform = [feature_name] if feature_name in numerical_features else []
            categorical_features_to_transform = [feature_name] if feature_name in categorical_features else []
            initial_depth = depths[feature_name]

            for level in range(initial_depth + 1):
                previous_X_feature_data = np.copy(X_test_transformed[:, indices])
                previous_y_transformed = np.copy(y_transformed)
                depths[feature_name] = initial_depth - level
                cls._transform_in_place(
                    dts=feature_dts,
                    X=X_test_transformed,
                    depths=depths,
                    numerical_features=numerical_features_to_transform,
                    categorical_features=categorical_features_to_transform,
                    feature_indices=feature_indices,
                    generalizations_arrays=generalization_arrays
                )
                y_transformed = estimator.predict(X_test_transformed)
                accuracy = accuracy_score(y_test, y_transformed)
                if accuracy < target_accuracy:
                    # TODO: Make sure level 1 actually does anything. The initial depth might be set too deep, although,
                    #  it does not affect correctness.
                    if level == 0:
                        untouched_features.append(feature_name)
                    else:
                        depths[feature_name] = initial_depth - level + 1
                    X_test_transformed[:, indices] = previous_X_feature_data
                    y_transformed = previous_y_transformed
                    break
        return untouched_features

    def _get_ordered_features(self, estimator, encoder, X_train, y_train, numerical_features, categorical_features,
                              feature_indices, random_state=None):
        return numerical_features + categorical_features

    def transform(self, X: pd.DataFrame):
        X_transformed = self._data_encoder.transform(X)
        self._transform_in_place(
            self._feature_dts, X_transformed,
            self._depths,
            self.generalizations["ranges"].keys(),
            self.generalizations["categories"].keys(),
            self._get_feature_indices(self._numerical_features, self._categorical_features,
                                      self._data_encoder.named_transformers_["cat"]),
            self._generalization_arrays
        )
        categorical_encoder = self._data_encoder.named_transformers_["cat"]
        X_out_cat = \
            categorical_encoder.inverse_transform(X_transformed[:, len(self._numerical_features):])
        return pd.DataFrame(np.concatenate([X_transformed[:, :len(self._numerical_features)], X_out_cat], axis=1),
                            columns=self._numerical_features + self._categorical_features)

    @property
    def generalizations(self):
        if self._generalizations is None:
            self._generalizations = self._get_generalizations_from_dts(
                self._feature_dts, self._numerical_features,
                self._categorical_features, self._untouched_features,
                self._categories_dict, self._generalization_arrays,
                self._depths
            )
        return self._generalizations

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
