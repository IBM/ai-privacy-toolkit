"""
This module implements all classes needed to perform data minimization
"""
from typing import Union
import pandas as pd
import numpy as np
import copy
import sys
from scipy.spatial import distance
from sklearn.base import BaseEstimator, TransformerMixin, MetaEstimatorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


class GeneralizeToRepresentative(BaseEstimator, MetaEstimatorMixin, TransformerMixin):
    """ A transformer that generalizes data to representative points.

    Learns data generalizations based on an original model's predictions
    and a target accuracy. Once the generalizations are learned, can
    receive one or more data records and transform them to representative
    points based on the learned generalization.

    An alternative way to use the transformer is to supply ``cells`` and
    ``features`` in init or set_params and those will be used to transform
    data to representatives. In this case, fit must still be called but
    there is no need to supply it with ``X`` and ``y``, and there is no
    need to supply an existing ``estimator`` to init.

    In summary, either ``estimator`` and ``target_accuracy`` should be
    supplied or ``cells`` and ``features`` should be supplied.

    Parameters
    ----------
    estimator : estimator, optional
        The original model for which generalization is being performed.
        Should be pre-fitted.

    target_accuracy : float, optional
        The required accuracy when applying the base model to the
        generalized data. Accuracy is measured relative to the original
        accuracy of the model.

    features : list of str, optional
        The feature names, in the order that they appear in the data.

    cells : list of object, optional
        The cells used to generalize records. Each cell must define a
        range or subset of categories for each feature, as well as a
        representative value for each feature.
        This parameter should be used when instantiating a transformer
        object without first fitting it.

    Attributes
    ----------
    cells_ : list of object
        The cells used to generalize records, as learned when calling fit.

    ncp_ : float
        The NCP (information loss) score of the resulting generalization,
        as measured on the training data.

    generalizations_ : object
        The generalizations that were learned (actual feature ranges).

    Notes
    -----


    """

    def __init__(self, estimator=None, target_accuracy=0.998, features=None,
                 cells=None, categorical_features=None):
        self.estimator = estimator
        self.target_accuracy = target_accuracy
        self.features = features
        self.cells = cells
        self.categorical_features = categorical_features

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and contained
            subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        ret = {}
        ret['target_accuracy'] = self.target_accuracy
        if deep:
            ret['features'] = copy.deepcopy(self.features)
            ret['cells'] = copy.deepcopy(self.cells)
            ret['estimator'] = self.estimator
        else:
            ret['features'] = copy.copy(self.features)
            ret['cells'] = copy.copy(self.cells)
        return ret

    def set_params(self, **params):
        """Set the parameters of this estimator.

        Returns
        -------
        self : object
            Returns self.
        """
        if 'target_accuracy' in params:
            self.target_accuracy = params['target_accuracy']
        if 'features' in params:
            self.features = params['features']
        if 'cells' in params:
            self.cells = params['cells']
        return self

    @property
    def generalizations(self):
        return self.generalizations_

    def fit_transform(self, X=None, y=None):
        """Learns the generalizations based on training data, and applies them to the data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features), optional
            The training input samples.
        y : array-like, shape (n_samples,), optional
            The target values. An array of int.
            This should contain the predictions of the original model on ``X``.

        Returns
        -------
        self : object
            Returns self.
        """
        self.fit(X, y)
        return self.transform(X)

    def fit(self, X: Union[np.ndarray, pd.DataFrame] = None, y=None):
        """Learns the generalizations based on training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features), optional
            The training input samples.
        y : array-like, shape (n_samples,), optional
            The target values. An array of int.
            This should contain the predictions of the original model on ``X``.

        Returns
        -------
        X_transformed : ndarray, shape (n_samples, n_features)
            The array containing the representative values to which each record in
            ``X`` is mapped.
        """

        # take into account that estimator, X, y, cells, features may be None

        if X is not None and y is not None:
            # X, y = check_X_y(X, y, accept_sparse=True)
            self.n_features_ = X.shape[1]
        elif self.features:
            self.n_features_ = len(self.features)
        else:
            self.n_features_ = 0

        if self.features:
            self._features = self.features
        # if features is None, use numbers instead of names
        elif self.n_features_ != 0:
            self._features = [i for i in range(self.n_features_)]
        else:
            self._features = None

        if self.cells:
            self.cells_ = self.cells
        else:
            self.cells_ = {}
        self.categoricalValues = {}

        # Going to fit
        # (currently not dealing with option to fit with only X and y and no estimator)
        if self.estimator and X is not None and y is not None:
            # divide dataset into train and test
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                                test_size=0.4,
                                                                random_state=18)

            # collect feature data (such as min, max)
            if type(X) == np.ndarray:
                train_data = pd.DataFrame(X_train, columns=self._features)
            else:
                train_data = X_train
            feature_data = {}
            for feature in self._features:
                if feature not in feature_data.keys():
                    fd = {}
                    values = list(train_data.loc[:, feature])
                    if feature not in self.categorical_features:
                        fd['min'] = min(values)
                        fd['max'] = max(values)
                        # fd['range'] = max(values) - min(values)
                    else:
                        fd['range'] = len(values)
                    feature_data[feature] = fd

            # prepare data for DT
            categorical_features = list(self.categorical_features)
            numeric_transformer = Pipeline(
                steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing'))]
            )

            # numeric_features = list(self._features) - list(self.categorical_features)
            numeric_features = [item for item in self._features if item not in self.categorical_features]
            categorical_transformer = OneHotEncoder(handle_unknown="ignore")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, numeric_features),
                    ("cat", categorical_transformer, categorical_features),
                ]
            )

            x_prepared = preprocessor.fit_transform(X_train)
            self.preprocessor = preprocessor
            x_prepared = x_prepared.astype(float)
            self.cells_ = {}
            self.dt_ = DecisionTreeClassifier(random_state=0, min_samples_split=2,
                                              min_samples_leaf=1)
            self.dt_.fit(x_prepared, y_train)
            self._modify_categorical_features(X)
            x_prepared = pd.DataFrame(x_prepared, columns=self.categorical_data.columns)

            self._calculate_cells()
            self._modify_cells()
            nodes = self._get_nodes_level(0)
            self._attach_cells_representatives(x_prepared, X_train, y_train, nodes)
            # self.cells_ currently holds the generalization created from the tree leaves
            self._calculate_generalizations()

            # apply generalizations to test data
            x_prepared_test = preprocessor.fit_transform(X_test)
            x_prepared_test = pd.DataFrame(x_prepared_test, columns=self.categorical_data.columns)

            generalized = self._generalize(x_prepared_test, nodes, self.cells_, self.cells_by_id_)

            # check accuracy
            accuracy = self.estimator.score(generalized, y_test)
            print('Initial accuracy of model on generalized data, relative to original model predictions '
                  '(base generalization derived from tree, before improvements): %f' % accuracy)

            # if accuracy above threshold, improve generalization
            if accuracy > self.target_accuracy:
                print('Improving generalizations')
                level = 1
                while accuracy > self.target_accuracy:
                    nodes = self._get_nodes_level(level)
                    self._calculate_level_cells(level)
                    self._attach_cells_representatives(x_prepared, X_train, y_train, nodes)
                    self._calculate_generalizations()
                    generalized = self._generalize(x_prepared_test, nodes, self.cells_,
                                                   self.cells_by_id_)
                    accuracy = self.estimator.score(generalized, y_test)
                    print('Pruned tree to level: %d, new relative accuracy: %f' % (level, accuracy))
                    level += 1

            # if accuracy below threshold, improve accuracy by removing features from generalization
            if accuracy < self.target_accuracy:
                print('Improving accuracy')
                while accuracy < self.target_accuracy:
                    y = X['age'][3]
                    removed_feature = self._remove_feature_from_generalization(x_prepared_test,
                                                                               nodes, y_test,
                                                                               feature_data, accuracy)
                    if removed_feature is None:
                        break

                    self._calculate_generalizations()
                    generalized = self._generalize(X_test, nodes, self.cells_, self.cells_by_id_)
                    accuracy = self.estimator.score(generalized, y_test)
                    print('Removed feature: %s, new relative accuracy: %f' % (removed_feature, accuracy))

            # self.cells_ currently holds the chosen generalization based on target accuracy

            # calculate iLoss
            self.ncp_ = self._calculate_ncp(X_test, self.generalizations_, feature_data)

        # Return the transformer
        return self

    def transform(self, X):
        """ Transforms data records to representative points.

        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_transformed : ndarray, shape (n_samples, n_features)
            The array containing the representative values to which each record in
            ``X`` is mapped.
        """

        # Check if fit has been called
        msg = 'This %(name)s instance is not initialized yet. ' \
              'Call ‘fit’ or ‘set_params’ with ' \
              'appropriate arguments before using this method.'
        check_is_fitted(self, ['cells', 'features'], msg=msg)

        # Input validation
        X = check_array(X, accept_sparse=True)
        if X.shape[1] != self.n_features_ and self.n_features_ != 0:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')

        if not self._features:
            self._features = [i for i in range(X.shape[1])]

        representatives = pd.DataFrame(columns=self._features)  # only columns
        generalized = pd.DataFrame(X, columns=self._features, copy=True)  # original data
        mapped = np.zeros(X.shape[0])  # to mark records we already mapped

        # iterate over cells (leaves in decision tree)
        for i in range(len(self.cells_)):
            # Copy the representatives from the cells into another data structure:
            # iterate over features in test data
            for feature in self._features:
                # if feature has a representative value in the cell and should not
                # be left untouched, take the representative value
                if feature in self.cells_[i]['representative'] and \
                        ('untouched' not in self.cells_[i]
                         or feature not in self.cells_[i]['untouched']):
                    representatives.loc[i, feature] = self.cells_[i]['representative'][feature]
                # else, drop the feature (removes from representatives columns that
                # do not have a representative value or should remain untouched)
                elif feature in representatives.columns.tolist():
                    representatives = representatives.drop(feature, axis=1)

            # get the indexes of all records that map to this cell
            indexes = self._get_record_indexes_for_cell(X, self.cells_[i], mapped)

            # replace the values in the representative columns with the representative
            # values (leaves others untouched)
            if indexes and not representatives.columns.empty:
                if len(indexes) > 1:
                    replace = pd.concat([representatives.loc[i].to_frame().T] * len(indexes)).reset_index(drop=True)
                else:
                    replace = representatives.loc[i].to_frame().T.reset_index(drop=True)
                replace.index = indexes
                generalized.loc[indexes, representatives.columns] = replace

        return generalized.to_numpy()

    def _get_record_indexes_for_cell(self, X, cell, mapped):
        return [i for i, x in enumerate(X) if not mapped.item(i) and
                self._cell_contains(cell, x, i, mapped)]

    def _cell_contains(self, cell, x, i, mapped):
        for f in self._features:
            if f in cell['ranges']:
                if not self._cell_contains_numeric(f, cell['ranges'][f], x):
                    return False
            else:
                # TODO: exception - feature not defined
                pass
        # Mark as mapped
        mapped.itemset(i, 1)
        return True

    def _modify_categorical_features(self, X):
        self.categoricalValues = {}
        self.oneHotVectorFeaturesToFeatures = {}
        featuresToRemove = []
        # print(self.data.columns)
        for feature in self.categorical_features:
            try:
                allValues = X.loc[:, feature]
                values = list(allValues.unique())
                self.categoricalValues[feature] = values
                X[feature] = pd.Categorical(X.loc[:, feature], categories=values, ordered=False)
                oneHotVector = pd.get_dummies(X[feature], prefix=feature)
                for oneHotVectorFeature in oneHotVector.columns:
                    self.oneHotVectorFeaturesToFeatures[oneHotVectorFeature] = feature
                X = pd.concat([X, oneHotVector], axis=1)
                featuresToRemove.append(feature)
            except KeyError:
                print('feature not found: ' + feature)
        self.categorical_data = X.drop(featuresToRemove, axis=1)
        # print(self.data.columns)

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

    def _calculate_cells(self):
        self.cells_by_id_ = {}
        self.cells_ = self._calculate_cells_recursive(0)

    def _calculate_cells_recursive(self, node):
        feature_index = self.dt_.tree_.feature[node]
        if feature_index == -2:
            # this is a leaf
            label = self._calculate_cell_label(node)
            hist = [int(i) for i in self.dt_.tree_.value[node][0]]
            cell = {'label': label, 'hist': hist, 'ranges': {}, 'id': int(node)}
            return [cell]

        cells = []
        feature = self.categorical_data.columns[feature_index]
        threshold = self.dt_.tree_.threshold[node]
        left_child = self.dt_.tree_.children_left[node]
        right_child = self.dt_.tree_.children_right[node]

        left_child_cells = self._calculate_cells_recursive(left_child)
        for cell in left_child_cells:
            if feature not in cell['ranges'].keys():
                cell['ranges'][feature] = {'start': None, 'end': None}
            if cell['ranges'][feature]['end'] is None:
                cell['ranges'][feature]['end'] = threshold
            cells.append(cell)
            self.cells_by_id_[cell['id']] = cell

        right_child_cells = self._calculate_cells_recursive(right_child)
        for cell in right_child_cells:
            if feature not in cell['ranges'].keys():
                cell['ranges'][feature] = {'start': None, 'end': None}
            if cell['ranges'][feature]['start'] is None:
                cell['ranges'][feature]['start'] = threshold
            cells.append(cell)
            self.cells_by_id_[cell['id']] = cell

        return cells

    def _calculate_cell_label(self, node):
        label_hist = self.dt_.tree_.value[node][0]
        return int(self.dt_.classes_[np.argmax(label_hist)])

    def _modify_cells(self):
        cells = []
        features = self.categorical_data.columns
        for cell in self.cells_:
            newCell = {'id': cell['id'], 'label': cell['label'], 'ranges': {}, 'categories': {}, 'hist': cell['hist'],
                       'representative': None}
            for feature in features:
                if feature in self.oneHotVectorFeaturesToFeatures.keys():
                    # feature is categorical and should be mapped
                    categoricalFeature = self.oneHotVectorFeaturesToFeatures[feature]
                    if categoricalFeature not in newCell['categories'].keys():
                        newCell['categories'][categoricalFeature] = self.categoricalValues[categoricalFeature].copy()
                    if feature in cell['ranges'].keys():
                        categoricalValue = feature[len(categoricalFeature) + 1:]
                        if cell['ranges'][feature]['start'] is not None:
                            # categorical feature must have this value
                            newCell['categories'][categoricalFeature] = [categoricalValue]
                        else:
                            # categorical feature can not have this value
                            if categoricalValue in newCell['categories'][categoricalFeature]:
                                newCell['categories'][categoricalFeature].remove(categoricalValue)
                else:
                    if feature in cell['ranges'].keys():
                        newCell['ranges'][feature] = cell['ranges'][feature]
                    else:
                        newCell['ranges'][feature] = {'start': None, 'end': None}
            cells.append(newCell)
            self.cells_by_id_[newCell['id']] = newCell
        self.cells_ = cells

    def _calculate_level_cells(self, level):
        if level < 0 or level > self.dt_.get_depth():
            # TODO: exception 'Illegal level %d' % level
            pass

        if level > 0:
            new_cells = []
            new_cells_by_id = {}
            nodes = self._get_nodes_level(level)
            if nodes:
                for node in nodes:
                    if self.dt_.tree_.feature[node] == -2:  # leaf node
                        new_cell = self.cells_by_id_[node]
                    else:
                        left_child = self.dt_.tree_.children_left[node]
                        right_child = self.dt_.tree_.children_right[node]
                        left_cell = self.cells_by_id_[left_child]
                        right_cell = self.cells_by_id_[right_child]
                        new_cell = {'id': int(node), 'ranges': {}, 'categories': {},
                                    'label': None, 'representative': None}
                        for feature in left_cell['ranges'].keys():
                            new_cell['ranges'][feature] = {}
                            new_cell['ranges'][feature]['start'] = left_cell['ranges'][feature]['start']
                            new_cell['ranges'][feature]['end'] = right_cell['ranges'][feature]['start']
                        for feature in left_cell['categories'].keys():
                            new_cell['categories'][feature] = \
                                list(set(left_cell['categories'][feature]) |
                                     set(right_cell['categories'][feature]))
                        self._calculate_level_cell_label(left_cell, right_cell, new_cell)
                    new_cells.append(new_cell)
                    new_cells_by_id[new_cell['id']] = new_cell
                self.cells_ = new_cells
                self.cells_by_id_ = new_cells_by_id
            # else: nothing to do, stay with previous cells

    def _calculate_level_cell_label(self, left_cell, right_cell, new_cell):
        new_cell['hist'] = [x + y for x, y in zip(left_cell['hist'], right_cell['hist'])]
        new_cell['label'] = int(self.dt_.classes_[np.argmax(new_cell['hist'])])

    def _get_nodes_level(self, level):
        # level = distance from lowest leaf
        node_depth = np.zeros(shape=self.dt_.tree_.node_count, dtype=np.int64)
        is_leaves = np.zeros(shape=self.dt_.tree_.node_count, dtype=bool)
        stack = [(0, -1)]  # seed is the root node id and its parent depth
        while len(stack) > 0:
            node_id, parent_depth = stack.pop()
            # depth = distance from root
            node_depth[node_id] = parent_depth + 1

            if self.dt_.tree_.children_left[node_id] != self.dt_.tree_.children_right[node_id]:
                stack.append((self.dt_.tree_.children_left[node_id], parent_depth + 1))
                stack.append((self.dt_.tree_.children_right[node_id], parent_depth + 1))
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

    def _attach_cells_representatives(self, samples, originalTrainFeatures, labelFeature, level_nodes):
        nodeIds = self._find_sample_nodes(samples, level_nodes)
        labels_df = pd.DataFrame(labelFeature, columns=['label'])
        for cell in self.cells_:
            cell['representative'] = {}
            # get all rows in cell
            indexes = [i for i, x in enumerate(nodeIds) if x == cell['id']]
            originalRows = originalTrainFeatures.iloc[indexes]
            sampleRows = samples.iloc[indexes]
            sample_labels = labels_df.iloc[indexes]['label'].values.tolist()
            # get rows with matching label
            indexes = [i for i, label in enumerate(sample_labels) if label == cell['label']]
            # matchSamples = sampleRows.loc[sampleRows[labelFeature] == cell['label']].drop([labelFeature], axis=1)
            matchSamples = sampleRows.iloc[indexes]
            matchRows = originalRows.iloc[indexes]
            # find the "middle" of the cluster
            array = matchSamples.values
            # Only works with numpy 1.9.0 and higher!!!
            # median = np.percentile(array, 50, interpolation='nearest', axis=0)
            median = np.median(array, axis=0)
            i = 0
            min = len(array)
            # print('array length %d' % min)
            min_dist = float("inf")
            for row in array:
                dist = distance.euclidean(row, median)
                if dist < min_dist:
                    min_dist = dist
                    min = i
                i = i + 1
            # print('min = %d' % min)
            row = matchRows.iloc[min]
            for feature in cell['ranges'].keys():
                cell['representative'][feature] = row[feature]
            for feature in cell['categories'].keys():
                cell['representative'][feature] = row[feature]

    def _find_sample_nodes(self, samples, nodes):
        paths = self.dt_.decision_path(samples).toarray()
        nodeSet = set(nodes)
        return [(list(set([i for i, v in enumerate(p) if v == 1]) & nodeSet))[0] for p in paths]

    def _generalize(self, data,  level_nodes, cells, cells_by_id):
        representatives = pd.DataFrame(columns=self.features)  # empty except for columns
        generalized = pd.DataFrame(data, columns=self.categorical_data.columns, copy=True)
        mapping_to_cells = self._map_to_cells(generalized, level_nodes, cells_by_id)
        # iterate over cells (leaves in decision tree)
        for i in range(len(cells)):
            representative_columns = list()
            # This code just copies the representatives from the cells into another data structure
            # iterate over features
            for feature in self.features:
                # if feature has a representative value in the cell and should not be left untouched,
                # take the representative value
                if feature in cells[i]['representative'] and ('untouched' not in cells[i] or
                                                              feature not in cells[i]['untouched']):
                    representatives.loc[i, feature] = cells[i]['representative'][feature]
                # else, drop the feature (removes from representatives columns that do not have a
                # representative value or should remain untouched)
                elif feature in representatives.columns.tolist():
                    representatives = representatives.drop(feature, axis=1)


                ###########################
                # map representative columns
                # representative_columns = list(self.categorical_data.columns)
                # for item in representative_columns:
                #    if item in self.categorical_features:
                #        tmp = self.oneHotVectorFeaturesToFeatures[item]
                #    else:
                #        tmp = item
                #    if tmp not in representatives.columns.tolist():
                #        representative_columns.remove(item)
                #############################


            # get the indexes of all records that map to this cell
            indexes = [j for j in range(len(mapping_to_cells)) if mapping_to_cells[j]['id'] == cells[i]['id']]
            # replaces the values in the representative columns with the representative values
            # (leaves others untouched)
            if indexes and not representatives.columns.empty:
                if len(indexes) > 1:
                    replace = pd.concat([representatives.loc[i].to_frame().T] * len(indexes)).reset_index(drop=True)
                else:
                    replace = representatives.loc[i].to_frame().T.reset_index(drop=True)
                replace.index = indexes
                replace = self.preprocessor.transform(replace)
                replace = pd.DataFrame(replace, indexes, columns=self.categorical_data.columns)
                generalized.loc[indexes, representative_columns] = replace

        return generalized

    def _map_to_cells(self, samples, nodes, cells_by_id):
        mapping_to_cells = []
        for index, row in samples.iterrows():
            cell = self._find_sample_cells([row], nodes, cells_by_id)[0]
            mapping_to_cells.append(cell)
        return mapping_to_cells

    def _find_sample_cells(self, samples, nodes, cells_by_id):
        node_ids = self._find_sample_nodes(samples, nodes)
        return [cells_by_id[nodeId] for nodeId in node_ids]

    def _remove_feature_from_generalization(self, samples, nodes, labels, feature_data, current_accuracy):
        feature = self._get_feature_to_remove(samples, nodes, labels, feature_data, current_accuracy)
        if feature is None:
            return None
        GeneralizeToRepresentative._remove_feature_from_cells(self.cells_, self.cells_by_id_, feature)
        # del self.generalizations_['ranges'][feature]
        # self.generalizations_['untouched'].append(feature)
        return feature

    def _get_feature_to_remove(self, samples, nodes, labels, feature_data, current_accuracy):
        # We want to remove features with low iLoss (NCP) and high accuracy gain
        # (after removing them)
        ranges = self.generalizations_['ranges']
        range_counts = self._find_range_count(samples, ranges)
        total = samples.size
        range_min = sys.float_info.max
        remove_feature = None

        for feature in ranges.keys():
            if feature not in self.generalizations_['untouched']:
                feature_ncp = self._calc_ncp_numeric(ranges[feature],
                                                     range_counts[feature],
                                                     feature_data[feature],
                                                     total)
                if feature_ncp > 0:
                    # divide by accuracy gain
                    new_cells = copy.deepcopy(self.cells_)
                    cells_by_id = copy.deepcopy(self.cells_by_id_)
                    GeneralizeToRepresentative._remove_feature_from_cells(new_cells, cells_by_id, feature)
                    generalized = self._generalize(samples, nodes, new_cells, cells_by_id)
                    accuracy_gain = self.estimator.score(generalized, labels) - current_accuracy
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
        self.generalizations_ = {'ranges': GeneralizeToRepresentative._calculate_ranges(self.cells_),
                                 'categories': GeneralizeToRepresentative._calculate_categories(self.cells_),
                                 'untouched': GeneralizeToRepresentative._calculate_untouched(self.cells_)}

    def _find_range_count(self, samples, ranges):
        samples_df = pd.DataFrame(samples, columns=self._features)
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
                    # range_counts[r].append(len(samples_df.loc[float(samples_df[r]) <= last_value]))
                    last_value = value
                # range_counts[r].append(len(samples_df.loc[float(samples_df[r]) > last_value]))
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
        categoryCounts = self._find_categories_count(samples, categories)

        total = samples.shape[0]
        total_ncp = 0
        total_features = len(generalizations['untouched'])
        for feature in ranges.keys():
            feature_ncp = self._calc_ncp_numeric(ranges[feature], range_counts[feature],
                                                 feature_data[feature], total)
            total_ncp = total_ncp + feature_ncp
            total_features += 1
        for feature in categories.keys():
            featureNCP = self._calc_ncp_categorical(categories[feature], categoryCounts[feature], feature_data[feature],
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
        categoricalFeaturesValues = GeneralizeToRepresentative._calculate_categorical_features_values(cells)
        for feature in categoricalFeaturesValues.keys():
            partitions = []
            values = categoricalFeaturesValues[feature]
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
        categoricalFeaturesValues = {}
        for cell in cells:
            for feature in [key for key in cell['categories'].keys() if
                            'untouched' not in cell or key not in cell['untouched']]:
                if feature not in categoricalFeaturesValues.keys():
                    categoricalFeaturesValues[feature] = []
                for value in cell['categories'][feature]:
                    if value not in categoricalFeaturesValues[feature]:
                        categoricalFeaturesValues[feature].append(value)
        return categoricalFeaturesValues

    @staticmethod
    def _are_inseparable(cells, feature, value1, value2):
        for cell in cells:
            if feature not in cell['categories'].keys():
                continue
            value1In = value1 in cell['categories'][feature]
            value2In = value2 in cell['categories'][feature]
            if value1In != value2In:
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
        categorySizes = [len(g) if len(g) > 1 else 0 for g in categories]
        normalizedCategorySizes = [s * n / total for s, n in zip(categorySizes, categoryCount)]
        averageGroupSize = sum(normalizedCategorySizes) / len(normalizedCategorySizes)
        return averageGroupSize / feature_data['range']  # number of values in category

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
            else:
                del cell['categories'][feature]
            cell['untouched'].append(feature)
            cells_by_id[cell['id']] = cell.copy()
