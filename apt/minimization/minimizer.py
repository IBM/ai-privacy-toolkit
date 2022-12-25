"""
This module implements all classes needed to perform data minimization
"""
from typing import Union, Optional
import pandas as pd
import numpy as np
import copy
import sys
from scipy.spatial import distance
from sklearn.base import BaseEstimator, TransformerMixin, MetaEstimatorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils.validation import check_is_fitted
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split

from apt.utils.datasets import ArrayDataset, DATA_PANDAS_NUMPY_TYPE
from apt.utils.models import Model, SklearnRegressor, ModelOutputType, SklearnClassifier


class GeneralizeToRepresentative(BaseEstimator, MetaEstimatorMixin, TransformerMixin):
    """
    A transformer that generalizes data to representative points.

    Learns data generalizations based on an original model's predictions
    and a target accuracy. Once the generalizations are learned, can
    receive one or more data records and transform them to representative
    points based on the learned generalization.
    An alternative way to use the transformer is to supply ``cells`` in
    init or set_params and those will be used to transform
    data to representatives. In this case, fit must still be called but
    there is no need to supply it with ``X`` and ``y``, and there is no
    need to supply an existing ``estimator`` to init.
    In summary, either ``estimator`` and ``target_accuracy`` should be
    supplied or ``cells`` should be supplied.

    :param estimator: The original model for which generalization is being performed. Should be pre-fitted.
    :type estimator: sklearn `BaseEstimator` or `Model`
    :param target_accuracy: The required relative accuracy when applying the base model to the generalized data.
                            Accuracy is measured relative to the original accuracy of the model.
    :type target_accuracy: float, optional
    :param cells: The cells used to generalize records. Each cell must define a range or subset of categories for
                  each feature, as well as a representative value for each feature. This parameter should be used
                  when instantiating a transformer object without first fitting it.
    :type cells: list of objects, optional
    :param categorical_features: The list of categorical features (if supplied, these featurtes will be one-hot
                                 encoded before using them to train the decision tree model).
    :param encoder: Optional encoder for encoding data before feeding it into the estimator (e.g., for categorical
                    features)
    :type encoder: sklearn OrdinalEncoder or OneHotEncoder
    :type categorical_features: list of strings, optional
    :param features_to_minimize: The features to be minimized.
    :type features_to_minimize: list of strings or int, optional
    :param train_only_features_to_minimize: Whether to train the tree just on the ``features_to_minimize`` or on all
                                            features. Default is only on ``features_to_minimize``.
    :type train_only_features_to_minimize: boolean, optional
    :param is_regression: Whether the model is a regression model or not (if False, assumes a classification model).
                          Default is False.
    :type is_regression: boolean, optional
    """

    def __init__(self, estimator: Union[BaseEstimator, Model] = None, target_accuracy: Optional[float] = 0.998,
                 cells: Optional[list] = None, categorical_features: Optional[Union[np.ndarray, list]] = None,
                 encoder: Optional[Union[OrdinalEncoder, OneHotEncoder]] = None,
                 features_to_minimize: Optional[Union[np.ndarray, list]] = None,
                 train_only_features_to_minimize: Optional[bool] = True,
                 is_regression: Optional[bool] = False):

        self.estimator = estimator
        if estimator is not None and not issubclass(estimator.__class__, Model):
            if is_regression:
                self.estimator = SklearnRegressor(estimator)
            else:
                self.estimator = SklearnClassifier(estimator, ModelOutputType.CLASSIFIER_PROBABILITIES)
        self.target_accuracy = target_accuracy
        self.cells = cells
        self.categorical_features = []
        if categorical_features:
            self.categorical_features = categorical_features
        self.features_to_minimize = features_to_minimize
        self.train_only_features_to_minimize = train_only_features_to_minimize
        self.is_regression = is_regression
        self.encoder = encoder

    def get_params(self, deep=True):
        """
        Get parameters

        :param deep: If True, will return the parameters for this estimator and contained
                     sub-objects that are estimators.
        :type deep: boolean, optional
        :return: Parameter names mapped to their values
        """
        ret = {}
        ret['target_accuracy'] = self.target_accuracy
        ret['categorical_features'] = self.categorical_features
        ret['features_to_minimize'] = self.features_to_minimize
        ret['train_only_features_to_minimize'] = self.train_only_features_to_minimize
        ret['is_regression'] = self.is_regression
        if deep:
            ret['cells'] = copy.deepcopy(self.cells)
            ret['estimator'] = self.estimator
            ret['encoder'] = self.encoder
        else:
            ret['cells'] = copy.copy(self.cells)
        return ret

    def set_params(self, **params):
        """
        Set parameters

        :param target_accuracy: The required relative accuracy when applying the base model to the generalized data.
                                Accuracy is measured relative to the original accuracy of the model.
        :type target_accuracy: float, optional
        :param cells: The cells used to generalize records. Each cell must define a range or subset of categories for
                      each feature, as well as a representative value for each feature. This parameter should be used
                      when instantiating a transformer object without first fitting it.
        :type cells: list of objects, optional
        :return: self
        """
        if 'target_accuracy' in params:
            self.target_accuracy = params['target_accuracy']
        if 'categorical_features' in params:
            self.categorical_features = params['categorical_features']
        if 'features_to_minimize' in params:
            self.features_to_minimize = params['features_to_minimize']
        if 'train_only_features_to_minimize' in params:
            self.train_only_features_to_minimize = params['train_only_features_to_minimize']
        if 'is_regression' in params:
            self.is_regression = params['is_regression']
        if 'cells' in params:
            self.cells = params['cells']
        return self

    @property
    def generalizations(self):
        """
        Return the generalizations derived from the model and test data.

        :return: generalizations object. Contains 3 sections: 'ranges' that contains ranges for numerical features,
                                 'categories' that contains sub-groups of categories for categorical features, and
                                 'untouched' that contains the features that could not be generalized.
        """
        return self._generalizations

    @property
    def ncp(self):
        """
        Return the NCP score of the generalizations.

        :return: ncp score as float.
        """
        return self._ncp

    def fit_transform(self, X: Optional[DATA_PANDAS_NUMPY_TYPE] = None, y: Optional[DATA_PANDAS_NUMPY_TYPE] = None,
                      features_names: Optional[list] = None, dataset: Optional[ArrayDataset] = None):
        """
        Learns the generalizations based on training data, and applies them to the data.

        :param X: The training input samples.
        :type X: {array-like, sparse matrix}, shape (n_samples, n_features), optional
        :param y: The target values. This should contain the predictions of the original model on ``X``.
        :type y: array-like, shape (n_samples,), optional
        :param features_names: The feature names, in the order that they appear in the data. Can be provided when
                               passing the data as ``X`` and ``y``
        :type features_names: list of strings, optional
        :param dataset: Data wrapper containing the training input samples and the predictions of the original model
                        on the training data. Either ``X``, ``y`` OR ``dataset`` need to be provided, not both.
        :type dataset: `ArrayDataset`, optional
        :return: Array containing the representative values to which each record in ``X`` is mapped, as numpy array or
                 pandas DataFrame (depending on the type of ``X``), shape (n_samples, n_features)
        """
        self.fit(X, y, features_names, dataset=dataset)
        return self.transform(X, features_names, dataset=dataset)

    def fit(self, X: Optional[DATA_PANDAS_NUMPY_TYPE] = None, y: Optional[DATA_PANDAS_NUMPY_TYPE] = None,
            features_names: Optional = None, dataset: ArrayDataset = None):
        """Learns the generalizations based on training data.

        :param X: The training input samples.
        :type X: {array-like, sparse matrix}, shape (n_samples, n_features), optional
        :param y: The target values. This should contain the predictions of the original model on ``X``.
        :type y: array-like, shape (n_samples,), optional
        :param features_names: The feature names, in the order that they appear in the data. Can be provided when
                               passing the data as ``X`` and ``y``
        :type features_names: list of strings, optional
        :param dataset: Data wrapper containing the training input samples and the predictions of the original model
                        on the training data. Either ``X``, ``y`` OR ``dataset`` need to be provided, not both.
        :type dataset: `ArrayDataset`, optional
        :return: self
        """

        # take into account that estimator, X, y, cells, features may be None
        if X is not None and y is not None:
            if dataset is not None:
                raise ValueError('Either X,y OR dataset need to be provided, not both')
            else:
                dataset = ArrayDataset(X, y, features_names)

        if dataset and dataset.get_samples() is not None and dataset.get_labels() is not None:
            self._n_features = dataset.get_samples().shape[1]
        elif dataset and dataset.features_names:
            self._n_features = len(dataset.features_names)
        else:
            self._n_features = 0

        if dataset and dataset.features_names:
            self._features = dataset.features_names
        # if features is None, use numbers instead of names
        elif self._n_features != 0:
            self._features = [str(i) for i in range(self._n_features)]
        else:
            self._features = None

        # Going to fit
        # (currently not dealing with option to fit with only X and y and no estimator)
        if self.estimator and dataset and dataset.get_samples() is not None and dataset.get_labels() is not None:
            x = pd.DataFrame(dataset.get_samples(), columns=self._features)
            if not self.features_to_minimize:
                self.features_to_minimize = self._features
            self.features_to_minimize = [str(i) for i in self.features_to_minimize]
            if not all(elem in self._features for elem in self.features_to_minimize):
                raise ValueError('features to minimize should be a subset of features names')
            x_QI = x.loc[:, self.features_to_minimize]

            # divide dataset into train and test
            used_data = x
            if self.train_only_features_to_minimize:
                used_data = x_QI
            if self.is_regression:
                X_train, X_test, y_train, y_test = train_test_split(x, dataset.get_labels(), test_size=0.4,
                                                                    random_state=14)
            else:
                try:
                    X_train, X_test, y_train, y_test = train_test_split(x, dataset.get_labels(),
                                                                        stratify=dataset.get_labels(), test_size=0.4,
                                                                        random_state=18)
                except ValueError:
                    print('Could not stratify split due to uncommon class value, doing unstratified split instead')
                    X_train, X_test, y_train, y_test = train_test_split(x, dataset.get_labels(), test_size=0.4,
                                                                        random_state=18)

            X_train_QI = X_train.loc[:, self.features_to_minimize]
            X_test_QI = X_test.loc[:, self.features_to_minimize]
            used_X_train = X_train
            used_X_test = X_test
            if self.train_only_features_to_minimize:
                used_X_train = X_train_QI
                used_X_test = X_test_QI

            # collect feature data (such as min, max)
            feature_data = {}
            for feature in self._features:
                if feature not in feature_data.keys():
                    fd = {}
                    values = list(x.loc[:, feature])
                    if feature not in self.categorical_features:
                        fd['min'] = min(values)
                        fd['max'] = max(values)
                        fd['range'] = max(values) - min(values)
                    else:
                        fd['range'] = len(np.unique(values))
                    feature_data[feature] = fd

            # default encoder in case none provided
            if self.encoder is None:
                numeric_features = [f for f in self._features if f not in self.categorical_features]
                numeric_transformer = Pipeline(
                    steps=[('imputer', SimpleImputer(strategy='constant', fill_value=0))]
                )
                categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)
                self.encoder = ColumnTransformer(
                    transformers=[
                        ("num", numeric_transformer, numeric_features),
                        ("cat", categorical_transformer, self.categorical_features),
                    ]
                )
                self.encoder.fit(x)

            self.cells = []
            self._categorical_values = {}

            if self.is_regression:
                self._dt = DecisionTreeRegressor(random_state=10, min_samples_split=2, min_samples_leaf=1)
            else:
                self._dt = DecisionTreeClassifier(random_state=0, min_samples_split=2,
                                                  min_samples_leaf=1)

            # prepare data for DT
            self._encode_categorical_features(used_data, save_mapping=True)
            x_prepared = self._encode_categorical_features(used_X_train)
            self._dt.fit(x_prepared, y_train)
            x_prepared_test = self._encode_categorical_features(used_X_test)

            self._calculate_cells()
            self._modify_cells()
            # features that are not from QI should not be part of generalizations
            for feature in self._features:
                if feature not in self.features_to_minimize:
                    self._remove_feature_from_cells(self.cells, self._cells_by_id, feature)

            nodes = self._get_nodes_level(0)
            self._attach_cells_representatives(x_prepared, used_X_train, y_train, nodes)

            # self._cells currently holds the generalization created from the tree leaves
            self._calculate_generalizations()
            generalized = self._generalize(X_test, x_prepared_test, nodes, self.cells, self._cells_by_id)

            # check accuracy
            accuracy = self.estimator.score(ArrayDataset(self.encoder.transform(generalized), y_test))
            print('Initial accuracy of model on generalized data, relative to original model predictions '
                  '(base generalization derived from tree, before improvements): %f' % accuracy)

            # if accuracy above threshold, improve generalization
            if accuracy > self.target_accuracy:
                print('Improving generalizations')
                level = 1
                while accuracy > self.target_accuracy:
                    cells_previous_iter = self.cells
                    generalization_prev_iter = self._generalizations
                    cells_by_id_prev = self._cells_by_id
                    nodes = self._get_nodes_level(level)

                    try:
                        self._calculate_level_cells(level)
                    except TypeError as e:
                        print(e)
                        break

                    self._attach_cells_representatives(x_prepared, used_X_train, y_train, nodes)

                    self._calculate_generalizations()
                    generalized = self._generalize(X_test, x_prepared_test, nodes, self.cells,
                                                   self._cells_by_id)
                    accuracy = self.estimator.score(ArrayDataset(self.encoder.transform(generalized), y_test))
                    # if accuracy passed threshold roll back to previous iteration generalizations
                    if accuracy < self.target_accuracy:
                        self.cells = cells_previous_iter
                        self._generalizations = generalization_prev_iter
                        self._cells_by_id = cells_by_id_prev
                        break
                    else:
                        print('Pruned tree to level: %d, new relative accuracy: %f' % (level, accuracy))
                        level += 1

            # if accuracy below threshold, improve accuracy by removing features from generalization
            elif accuracy < self.target_accuracy:
                print('Improving accuracy')
                while accuracy < self.target_accuracy:
                    removed_feature = self._remove_feature_from_generalization(X_test, x_prepared_test,
                                                                               nodes, y_test,
                                                                               feature_data, accuracy)
                    if removed_feature is None:
                        break

                    self._calculate_generalizations()
                    generalized = self._generalize(X_test, x_prepared_test, nodes, self.cells, self._cells_by_id)
                    accuracy = self.estimator.score(ArrayDataset(self.encoder.transform(generalized), y_test))
                    print('Removed feature: %s, new relative accuracy: %f' % (removed_feature, accuracy))

            # self._cells currently holds the chosen generalization based on target accuracy

            # calculate iLoss
            self._ncp = self._calculate_ncp(X_test, self._generalizations, feature_data)

        # Return the transformer
        return self

    def transform(self, X: Optional[DATA_PANDAS_NUMPY_TYPE] = None, features_names: Optional[list] = None,
                  dataset: Optional[ArrayDataset] = None):
        """ Transforms data records to representative points.

        :param X: The training input samples.
        :type X: {array-like, sparse matrix}, shape (n_samples, n_features), optional
        :param features_names: The feature names, in the order that they appear in the data. Can be provided when
                               passing the data as ``X`` and ``y``
        :type features_names: list of strings, optional
        :param dataset: Data wrapper containing the training input samples and the predictions of the original model
                        on the training data. Either ``X`` OR ``dataset`` need to be provided, not both.
        :type dataset: `ArrayDataset`, optional
        :return: Array containing the representative values to which each record in ``X`` is mapped, as numpy array or
                 pandas DataFrame (depending on the type of ``X``), shape (n_samples, n_features)
        """

        # Check if fit has been called
        msg = 'This %(name)s instance is not initialized yet. ' \
              'Call ‘fit’ or ‘set_params’ with ' \
              'appropriate arguments before using this method.'
        check_is_fitted(self, ['cells'], msg=msg)

        if X is not None:
            if dataset is not None:
                raise ValueError('Either X OR dataset need to be provided, not both')
            else:
                dataset = ArrayDataset(X, features_names=features_names)
        elif dataset is None:
            raise ValueError('Either X OR dataset need to be provided, not both')
        if dataset and dataset.features_names:
            self._features = dataset.features_names
        if dataset and dataset.get_samples() is not None:
            x = pd.DataFrame(dataset.get_samples(), columns=self._features)

        if x.shape[1] != self._n_features and self._n_features != 0:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')

        if not self._features:
            self._features = [i for i in range(x.shape[1])]

        mapped = np.zeros(x.shape[0])  # to mark records we already mapped
        all_indexes = []
        for i in range(len(self.cells)):
            indexes = self._get_record_indexes_for_cell(x, self.cells[i], mapped)
            all_indexes.append(indexes)
        generalized = self._generalize_indexes(x, self.cells, all_indexes)

        if dataset and dataset.is_pandas:
            return generalized
        elif isinstance(X, pd.DataFrame):
            return generalized
        return generalized.to_numpy()

    def _get_record_indexes_for_cell(self, X, cell, mapped):
        indexes = []
        for index, row in X.iterrows():
            if not mapped.item(index) and self._cell_contains(cell, row, index, mapped):
                indexes.append(index)
        return indexes

    def _cell_contains(self, cell, x, i, mapped):
        for f in self._features:
            if f in cell['ranges']:
                if not self._cell_contains_numeric(f, cell['ranges'][f], x):
                    return False
            elif f in cell['categories']:
                if not self._cell_contains_categorical(f, cell['categories'][f], x):
                    return False
            elif f in cell['untouched']:
                continue
            else:
                raise TypeError("feature " + f + "not found in cell" + cell['id'])
        # Mark as mapped
        mapped.itemset(i, 1)
        return True

    def _encode_categorical_features(self, X, save_mapping=False):
        if save_mapping:
            self._categorical_values = {}
            self._one_hot_vector_features_to_features = {}
        features_to_remove = []
        used_features = self._features
        if self.train_only_features_to_minimize:
            used_features = self.features_to_minimize
        for feature in self.categorical_features:
            if feature in used_features:
                try:
                    all_values = X.loc[:, feature]
                    values = list(all_values.unique())
                    if save_mapping:
                        self._categorical_values[feature] = values
                    X[feature] = pd.Categorical(X.loc[:, feature], categories=self._categorical_values[feature],
                                                ordered=False)
                    ohe = pd.get_dummies(X[feature], prefix=feature)
                    if save_mapping:
                        for one_hot_vector_feature in ohe.columns:
                            self._one_hot_vector_features_to_features[one_hot_vector_feature] = feature
                    X = pd.concat([X, ohe], axis=1)
                    features_to_remove.append(feature)
                except KeyError:
                    print("feature " + feature + "not found in training data")

        new_data = X.drop(features_to_remove, axis=1)
        if save_mapping:
            self._encoded_features = new_data.columns
        return new_data

    def _cell_contains_numeric(self, f, range, x):
        i = self._features.index(f)
        # convert x to ndarray to allow indexing
        a = np.array(x)
        value = a.item(i)
        if range['start']:
            if value <= range['start']:
                return False
        if range['end']:
            if value > range['end']:
                return False
        return True

    def _cell_contains_categorical(self, f, range, x):
        i = self._features.index(f)
        # convert x to ndarray to allow indexing
        a = np.array(x)
        value = a.item(i)
        if value in range:
            return True
        return False

    def _calculate_cells(self):
        self._cells_by_id = {}
        self.cells = self._calculate_cells_recursive(0)

    def _calculate_cells_recursive(self, node):
        feature_index = self._dt.tree_.feature[node]
        if feature_index == -2:
            # this is a leaf
            # if it is a regression problem we do not use label
            label = self._calculate_cell_label(node) if not self.is_regression else 1
            hist = [int(i) for i in self._dt.tree_.value[node][0]] if not self.is_regression else []
            cell = {'label': label, 'hist': hist, 'ranges': {}, 'id': int(node)}
            return [cell]

        cells = []
        feature = self._encoded_features[feature_index]
        threshold = self._dt.tree_.threshold[node]
        left_child = self._dt.tree_.children_left[node]
        right_child = self._dt.tree_.children_right[node]

        left_child_cells = self._calculate_cells_recursive(left_child)
        for cell in left_child_cells:
            if feature not in cell['ranges'].keys():
                cell['ranges'][feature] = {'start': None, 'end': None}
            if cell['ranges'][feature]['end'] is None:
                cell['ranges'][feature]['end'] = threshold
            cells.append(cell)
            self._cells_by_id[cell['id']] = cell

        right_child_cells = self._calculate_cells_recursive(right_child)
        for cell in right_child_cells:
            if feature not in cell['ranges'].keys():
                cell['ranges'][feature] = {'start': None, 'end': None}
            if cell['ranges'][feature]['start'] is None:
                cell['ranges'][feature]['start'] = threshold
            cells.append(cell)
            self._cells_by_id[cell['id']] = cell

        return cells

    def _calculate_cell_label(self, node):
        label_hist = self._dt.tree_.value[node][0]
        return int(self._dt.classes_[np.argmax(label_hist)])

    def _modify_cells(self):
        cells = []
        features = self._encoded_features
        for cell in self.cells:
            new_cell = {'id': cell['id'], 'label': cell['label'], 'ranges': {}, 'categories': {}, 'hist': cell['hist'],
                        'untouched': [], 'representative': None}
            for feature in features:
                if feature in self._one_hot_vector_features_to_features.keys():
                    # feature is categorical and should be mapped
                    categorical_feature = self._one_hot_vector_features_to_features[feature]
                    if categorical_feature not in new_cell['categories'].keys():
                        new_cell['categories'][categorical_feature] = self._categorical_values[
                            categorical_feature].copy()
                    if feature in cell['ranges'].keys():
                        categorical_value = feature[len(categorical_feature) + 1:]
                        if cell['ranges'][feature]['start'] is not None:
                            # categorical feature must have this value
                            new_cell['categories'][categorical_feature] = [categorical_value]
                        else:
                            # categorical feature can not have this value
                            if categorical_value in new_cell['categories'][categorical_feature]:
                                new_cell['categories'][categorical_feature].remove(categorical_value)
                else:
                    if feature in cell['ranges'].keys():
                        new_cell['ranges'][feature] = cell['ranges'][feature]
                    else:
                        new_cell['ranges'][feature] = {'start': None, 'end': None}
            cells.append(new_cell)
            self._cells_by_id[new_cell['id']] = new_cell
        self.cells = cells

    def _calculate_level_cells(self, level):
        if level < 0 or level > self._dt.get_depth():
            raise TypeError("Illegal level %d' % level", level)

        if level > 0:
            new_cells = []
            new_cells_by_id = {}
            nodes = self._get_nodes_level(level)
            if nodes:
                for node in nodes:
                    if self._dt.tree_.feature[node] == -2:  # leaf node
                        new_cell = self._cells_by_id[node]
                    else:
                        left_child = self._dt.tree_.children_left[node]
                        right_child = self._dt.tree_.children_right[node]
                        left_cell = self._cells_by_id[left_child]
                        right_cell = self._cells_by_id[right_child]
                        new_cell = {'id': int(node), 'ranges': {}, 'categories': {}, 'untouched': [],
                                    'label': None, 'representative': None}
                        for feature in left_cell['ranges'].keys():
                            new_cell['ranges'][feature] = {}
                            new_cell['ranges'][feature]['start'] = left_cell['ranges'][feature]['start']
                            new_cell['ranges'][feature]['end'] = right_cell['ranges'][feature]['start']
                        for feature in left_cell['categories'].keys():
                            new_cell['categories'][feature] = \
                                list(set(left_cell['categories'][feature])
                                     | set(right_cell['categories'][feature]))
                        for feature in left_cell['untouched']:
                            if feature in right_cell['untouched']:
                                new_cell['untouched'].append(feature)
                        self._calculate_level_cell_label(left_cell, right_cell, new_cell)
                    new_cells.append(new_cell)
                    new_cells_by_id[new_cell['id']] = new_cell
                self.cells = new_cells
                self._cells_by_id = new_cells_by_id
            # else: nothing to do, stay with previous cells

    def _calculate_level_cell_label(self, left_cell, right_cell, new_cell):
        new_cell['hist'] = [x + y for x, y in
                            zip(left_cell['hist'], right_cell['hist'])] if not self.is_regression else []
        new_cell['label'] = int(self._dt.classes_[np.argmax(new_cell['hist'])]) if not self.is_regression else 1

    def _get_nodes_level(self, level):
        # level = distance from lowest leaf
        node_depth = np.zeros(shape=self._dt.tree_.node_count, dtype=np.int64)
        is_leaves = np.zeros(shape=self._dt.tree_.node_count, dtype=bool)
        stack = [(0, -1)]  # seed is the root node id and its parent depth
        while len(stack) > 0:
            node_id, parent_depth = stack.pop()
            # depth = distance from root
            node_depth[node_id] = parent_depth + 1

            if self._dt.tree_.children_left[node_id] != self._dt.tree_.children_right[node_id]:
                stack.append((self._dt.tree_.children_left[node_id], parent_depth + 1))
                stack.append((self._dt.tree_.children_right[node_id], parent_depth + 1))
            else:
                is_leaves[node_id] = True

        # depth of entire tree
        max_depth = max(node_depth)
        # depth of current level
        depth = max_depth - level
        # level is higher than root
        if depth < 0:
            return None
        # return all nodes with depth == level or leaves higher than level
        return [i for i, x in enumerate(node_depth) if x == depth or (x < depth and is_leaves[i])]

    def _attach_cells_representatives(self, prepared_data, originalTrainFeatures, labelFeature, level_nodes):
        # prepared data include one hot encoded categorical data,
        # if there is no categorical data prepared data is original data
        nodeIds = self._find_sample_nodes(prepared_data, level_nodes)
        labels_df = pd.DataFrame(labelFeature, columns=['label'])
        for cell in self.cells:
            cell['representative'] = {}
            # get all rows in cell
            indexes = [i for i, x in enumerate(nodeIds) if x == cell['id']]
            original_rows = originalTrainFeatures.iloc[indexes]
            sample_rows = prepared_data.iloc[indexes]
            sample_labels = labels_df.iloc[indexes]['label'].values.tolist()
            # get rows with matching label
            if self.is_regression:
                match_samples = sample_rows
                match_rows = original_rows
            else:
                indexes = [i for i, label in enumerate(sample_labels) if label == cell['label']]
                match_samples = sample_rows.iloc[indexes]
                match_rows = original_rows.iloc[indexes]
            # find the "middle" of the cluster
            array = match_samples.values
            # Only works with numpy 1.9.0 and higher!!!
            median = np.median(array, axis=0)
            i = 0
            min = len(array)
            min_dist = float("inf")
            for row in array:
                dist = distance.euclidean(row, median)
                if dist < min_dist:
                    min_dist = dist
                    min = i
                i = i + 1
            row = match_rows.iloc[min]
            for feature in cell['ranges'].keys():
                cell['representative'][feature] = row[feature]
            for feature in cell['categories'].keys():
                cell['representative'][feature] = row[feature]

    def _find_sample_nodes(self, samples, nodes):
        paths = self._dt.decision_path(samples).toarray()
        nodeSet = set(nodes)
        return [(list(set([i for i, v in enumerate(p) if v == 1]) & nodeSet))[0] for p in paths]

    def _generalize(self, original_data, prepared_data, level_nodes, cells, cells_by_id):
        mapping_to_cells = self._map_to_cells(prepared_data, level_nodes, cells_by_id)
        all_indexes = []
        for i in range(len(cells)):
            # get the indexes of all records that map to this cell
            indexes = [j for j in mapping_to_cells if mapping_to_cells[j]['id'] == cells[i]['id']]
            all_indexes.append(indexes)
        return self._generalize_indexes(original_data, cells, all_indexes)

    def _generalize_indexes(self, original_data, cells, all_indexes):
        # prepared data include one hot encoded categorical data + QI
        representatives = pd.DataFrame(columns=self._features)  # empty except for columns
        original_data_generalized = pd.DataFrame(original_data, columns=self._features, copy=True)

        # iterate over cells (leaves in decision tree)
        for i in range(len(cells)):
            # This code just copies the representatives from the cells into another data structure
            # iterate over features
            for feature in self._features:
                # if feature has a representative value in the cell and should not be left untouched,
                # take the representative value
                if feature in cells[i]['representative'] \
                        and ('untouched' not in cells[i] or feature not in cells[i]['untouched']):
                    representatives.loc[i, feature] = cells[i]['representative'][feature]
                # else, drop the feature (removes from representatives columns that do not have a
                # representative value or should remain untouched)
                elif feature in representatives.columns.tolist():
                    representatives = representatives.drop(feature, axis=1)

            indexes = all_indexes[i]
            # replaces the values in the representative columns with the representative values
            # (leaves others untouched)
            if indexes and not representatives.columns.empty:
                if len(indexes) > 1:
                    replace = pd.concat([representatives.loc[i].to_frame().T] * len(indexes)).reset_index(drop=True)
                else:
                    replace = representatives.loc[i].to_frame().T.reset_index(drop=True)
                replace.index = indexes
                replace = pd.DataFrame(replace, indexes, columns=self._features)
                original_data_generalized.loc[indexes, representatives.columns.tolist()] = replace

        return original_data_generalized

    def _map_to_cells(self, samples, nodes, cells_by_id):
        mapping_to_cells = {}
        for index, row in samples.iterrows():
            cell = self._find_sample_cells([row], nodes, cells_by_id)[0]
            mapping_to_cells[index] = cell
        return mapping_to_cells

    def _find_sample_cells(self, samples, nodes, cells_by_id):
        node_ids = self._find_sample_nodes(samples, nodes)
        return [cells_by_id[nodeId] for nodeId in node_ids]

    def _remove_feature_from_generalization(self, original_data, prepared_data, nodes, labels, feature_data,
                                            current_accuracy):
        # prepared data include one hot encoded categorical data,
        # if there is no categorical data prepared data is original data
        feature = self._get_feature_to_remove(original_data, prepared_data, nodes, labels, feature_data,
                                              current_accuracy)
        if feature is None:
            return None
        GeneralizeToRepresentative._remove_feature_from_cells(self.cells, self._cells_by_id, feature)
        return feature

    def _get_feature_to_remove(self, original_data, prepared_data, nodes, labels, feature_data, current_accuracy):
        # prepared data include one hot encoded categorical data,
        # if there is no categorical data prepared data is original data
        # We want to remove features with low iLoss (NCP) and high accuracy gain
        # (after removing them)
        ranges = self._generalizations['ranges']
        range_counts = self._find_range_count(original_data, ranges)
        total = prepared_data.size
        range_min = sys.float_info.max
        remove_feature = None
        categories = self.generalizations['categories']
        category_counts = self._find_categories_count(original_data, categories)

        for feature in ranges.keys():
            if feature not in self._generalizations['untouched']:
                feature_ncp = self._calc_ncp_numeric(ranges[feature],
                                                     range_counts[feature],
                                                     feature_data[feature],
                                                     total)
                if feature_ncp > 0:
                    # divide by accuracy gain
                    new_cells = copy.deepcopy(self.cells)
                    cells_by_id = copy.deepcopy(self._cells_by_id)
                    GeneralizeToRepresentative._remove_feature_from_cells(new_cells, cells_by_id, feature)
                    generalized = self._generalize(original_data, prepared_data, nodes, new_cells, cells_by_id)
                    accuracy_gain = self.estimator.score(ArrayDataset(self.encoder.transform(generalized),
                                                                      labels)) - current_accuracy
                    if accuracy_gain < 0:
                        accuracy_gain = 0
                    if accuracy_gain != 0:
                        feature_ncp = feature_ncp / accuracy_gain

                if feature_ncp < range_min:
                    range_min = feature_ncp
                    remove_feature = feature

        for feature in categories.keys():
            if feature not in self.generalizations['untouched']:
                feature_ncp = self._calc_ncp_categorical(categories[feature],
                                                         category_counts[feature],
                                                         feature_data[feature],
                                                         total)
                if feature_ncp > 0:
                    # divide by accuracy loss
                    new_cells = copy.deepcopy(self.cells)
                    cells_by_id = copy.deepcopy(self._cells_by_id)
                    GeneralizeToRepresentative._remove_feature_from_cells(new_cells, cells_by_id, feature)
                    generalized = self._generalize(original_data, prepared_data, nodes, new_cells, cells_by_id)
                    accuracy_gain = self.estimator.score(ArrayDataset(self.encoder.transform(generalized),
                                                                      labels)) - current_accuracy

                    if accuracy_gain < 0:
                        accuracy_gain = 0
                    if accuracy_gain != 0:
                        feature_ncp = feature_ncp / accuracy_gain
                if feature_ncp < range_min:
                    range_min = feature_ncp
                    remove_feature = feature

        print('feature to remove: ' + (str(remove_feature) if remove_feature is not None else 'none'))
        return remove_feature

    def _calculate_generalizations(self):
        self._generalizations = {'ranges': GeneralizeToRepresentative._calculate_ranges(self.cells),
                                 'categories': GeneralizeToRepresentative._calculate_categories(self.cells),
                                 'untouched': GeneralizeToRepresentative._calculate_untouched(self.cells)}
        self._remove_categorical_untouched(self._generalizations)

    def _find_range_count(self, samples, ranges):
        samples_df = pd.DataFrame(samples, columns=self._encoded_features)
        range_counts = {}
        last_value = None
        for r in ranges.keys():
            range_counts[r] = []
            # if empty list, all samples should be counted
            if not ranges[r]:
                range_counts[r].append(samples_df.shape[0])
            else:
                for value in ranges[r]:
                    counter = [item for item in samples_df[r] if int(item) <= value]
                    range_counts[r].append(len(counter))
                    last_value = value
                counter = [item for item in samples_df[r] if int(item) <= last_value]
                range_counts[r].append(len(counter))
        return range_counts

    def _find_categories_count(self, samples, categories):
        category_counts = {}
        for c in categories.keys():
            category_counts[c] = []
            for value in categories[c]:
                category_counts[c].append(len(samples.loc[samples[c].isin(value)]))
        return category_counts

    def _calculate_ncp(self, samples, generalizations, feature_data):
        # supressed features are already taken care of within _calc_ncp_numeric
        ranges = generalizations['ranges']
        categories = generalizations['categories']
        range_counts = self._find_range_count(samples, ranges)
        category_counts = self._find_categories_count(samples, categories)

        total = samples.shape[0]
        total_ncp = 0
        total_features = len(generalizations['untouched'])
        for feature in ranges.keys():
            feature_ncp = self._calc_ncp_numeric(ranges[feature], range_counts[feature],
                                                 feature_data[feature], total)
            total_ncp = total_ncp + feature_ncp
            total_features += 1
        for feature in categories.keys():
            featureNCP = self._calc_ncp_categorical(categories[feature], category_counts[feature],
                                                    feature_data[feature],
                                                    total)
            total_ncp = total_ncp + featureNCP
            total_features += 1
        if total_features == 0:
            return 0
        return total_ncp / total_features

    @staticmethod
    def _calculate_ranges(cells):
        ranges = {}
        for cell in cells:
            for feature in [key for key in cell['ranges'].keys() if
                            'untouched' not in cell or key not in cell['untouched']]:
                if feature not in ranges.keys():
                    ranges[feature] = []
                if cell['ranges'][feature]['start'] is not None:
                    ranges[feature].append(cell['ranges'][feature]['start'])
                if cell['ranges'][feature]['end'] is not None:
                    ranges[feature].append(cell['ranges'][feature]['end'])
        for feature in ranges.keys():
            ranges[feature] = list(set(ranges[feature]))
            ranges[feature].sort()
        return ranges

    @staticmethod
    def _calculate_categories(cells):
        categories = {}
        categorical_features_values = GeneralizeToRepresentative._calculate_categorical_features_values(cells)
        for feature in categorical_features_values.keys():
            partitions = []
            values = categorical_features_values[feature]
            assigned = []
            for i in range(len(values)):
                value1 = values[i]
                if value1 in assigned:
                    continue
                partition = [value1]
                assigned.append(value1)
                for j in range(len(values)):
                    if j <= i:
                        continue
                    value2 = values[j]
                    if GeneralizeToRepresentative._are_inseparable(cells, feature, value1, value2):
                        partition.append(value2)
                        assigned.append(value2)
                partitions.append(partition)
            categories[feature] = partitions
        return categories

    @staticmethod
    def _calculate_categorical_features_values(cells):
        categorical_features_values = {}
        for cell in cells:
            for feature in [key for key in cell['categories'].keys() if
                            'untouched' not in cell or key not in cell['untouched']]:
                if feature not in categorical_features_values.keys():
                    categorical_features_values[feature] = []
                for value in cell['categories'][feature]:
                    if value not in categorical_features_values[feature]:
                        categorical_features_values[feature].append(value)
        return categorical_features_values

    @staticmethod
    def _are_inseparable(cells, feature, value1, value2):
        for cell in cells:
            if feature not in cell['categories'].keys():
                continue
            value1_in = value1 in cell['categories'][feature]
            value2_in = value2 in cell['categories'][feature]
            if value1_in != value2_in:
                return False
        return True

    @staticmethod
    def _calculate_untouched(cells):
        untouched_lists = [cell['untouched'] if 'untouched' in cell else [] for cell in cells]
        untouched = set(untouched_lists[0])
        untouched = untouched.intersection(*untouched_lists)
        return list(untouched)

    @staticmethod
    def _calc_ncp_categorical(categories, categoryCount, feature_data, total):
        category_sizes = [len(g) if len(g) > 1 else 0 for g in categories]
        normalized_category_sizes = [s * n / total for s, n in zip(category_sizes, categoryCount)]
        average_group_size = sum(normalized_category_sizes) / len(normalized_category_sizes)
        return average_group_size / feature_data['range']  # number of values in category

    @staticmethod
    def _calc_ncp_numeric(feature_range, range_count, feature_data, total):
        # if there are no ranges, feature is supressed and iLoss is 1
        if not feature_range:
            return 1
        # range only contains the split values, need to add min and max value of feature
        # to enable computing sizes of all ranges
        new_range = [feature_data['min']] + feature_range + [feature_data['max']]
        range_sizes = [b - a for a, b in zip(new_range[::1], new_range[1::1])]
        normalized_range_sizes = [s * n / total for s, n in zip(range_sizes, range_count)]
        average_range_size = sum(normalized_range_sizes) / len(normalized_range_sizes)
        return average_range_size / (feature_data['max'] - feature_data['min'])

    @staticmethod
    def _remove_feature_from_cells(cells, cells_by_id, feature):
        for cell in cells:
            if 'untouched' not in cell:
                cell['untouched'] = []
            if feature in cell['ranges'].keys():
                del cell['ranges'][feature]
            elif feature in cell['categories'].keys():
                del cell['categories'][feature]
            cell['untouched'].append(feature)
            cells_by_id[cell['id']] = cell.copy()

    @staticmethod
    def _remove_categorical_untouched(generalizations):
        to_remove = []
        for feature in generalizations['categories'].keys():
            category_sizes = [len(g) if len(g) > 1 else 0 for g in generalizations['categories'][feature]]
            if sum(category_sizes) == 0:
                if 'untouched' not in generalizations:
                    generalizations['untouched'] = []
                generalizations['untouched'].append(feature)
                to_remove.append(feature)

        for feature in to_remove:
            del generalizations['categories'][feature]
