"""
This module implements all classes needed to perform data minimization
"""
from typing import Union, Optional
from dataclasses import dataclass
from collections import Counter
import pandas as pd
import numpy as np
import copy
import sys
from scipy.spatial import distance
from sklearn.base import BaseEstimator, TransformerMixin, MetaEstimatorMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils.validation import check_is_fitted
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split

from apt.utils.datasets import ArrayDataset, DATA_PANDAS_NUMPY_TYPE
from apt.utils.models import Model, SklearnRegressor, SklearnClassifier, \
    CLASSIFIER_SINGLE_OUTPUT_CLASS_PROBABILITIES


@dataclass
class NCPScores:
    fit_score: float = None
    transform_score: float = None
    generalizations_score: float = None


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
                    features). If not provided, the data will be fed as is directly to the estimator.
    :type encoder: sklearn OrdinalEncoder or OneHotEncoder
    :type categorical_features: list of strings or integers, optional
    :param features_to_minimize: The features to be minimized. If not provided, all features will be minimized.
    :type features_to_minimize: list of strings or int, optional
    :param feature_slices: If some of the features to be minimized represent 1-hot encoded features that need to remain
                           consistent after minimization, provide a list containing the list of column names
                           or indexes that represent a single feature.
    :type feature_slices: list of lists of strings or integers, optional
    :param train_only_features_to_minimize: Whether to train the tree just on the ``features_to_minimize`` or on all
                                            features. Default is only on ``features_to_minimize``.
    :type train_only_features_to_minimize: boolean, optional
    :param is_regression: Whether the model is a regression model or not (if False, assumes a classification model).
                          Default is False.
    :type is_regression: boolean, optional
    :param generalize_using_transform: Indicates how to calculate NCP and accuracy during the generalization
                                       process. True means that the `transform` method is used to transform original
                                       data into generalized data that is used for accuracy and NCP calculation.
                                       False indicates that the `generalizations` structure should be used.
                                       Default is True.
    :type generalize_using_transform: boolean, optional
    """

    def __init__(self, estimator: Union[BaseEstimator, Model] = None,
                 target_accuracy: Optional[float] = 0.998,
                 cells: Optional[list] = None,
                 categorical_features: Optional[Union[np.ndarray, list]] = None,
                 encoder: Optional[Union[OrdinalEncoder, OneHotEncoder]] = None,
                 features_to_minimize: Optional[Union[np.ndarray, list]] = None,
                 feature_slices: Optional[list] = None,
                 train_only_features_to_minimize: Optional[bool] = True,
                 is_regression: Optional[bool] = False,
                 generalize_using_transform: bool = True):

        self.estimator = estimator
        if estimator is not None and not issubclass(estimator.__class__, Model):
            if is_regression:
                self.estimator = SklearnRegressor(estimator)
            else:
                # model output type is not critical as it only affects computation of nb_classes, which is in any case
                # the same currently for single and multi output probabilities.
                self.estimator = SklearnClassifier(estimator,
                                                   CLASSIFIER_SINGLE_OUTPUT_CLASS_PROBABILITIES)
        self.target_accuracy = target_accuracy
        self.cells = cells
        self.categorical_features = []
        if categorical_features:
            self.categorical_features = categorical_features
        self.features_to_minimize = features_to_minimize
        self.feature_slices = feature_slices
        if self.feature_slices:
            self.all_one_hot_features = {str(feature) for encoded in self.feature_slices for feature in encoded}
        else:
            self.all_one_hot_features = set()
        self.train_only_features_to_minimize = train_only_features_to_minimize
        self.is_regression = is_regression
        self.encoder = encoder
        self.generalize_using_transform = generalize_using_transform
        self._ncp_scores = NCPScores()
        self._feature_data = None
        self._categorical_values = {}
        self._dt = None
        self._features = None
        self._level = 0
        if cells:
            self._calculate_generalizations()

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
        ret['feature_slices'] = self.feature_slices
        ret['train_only_features_to_minimize'] = self.train_only_features_to_minimize
        ret['is_regression'] = self.is_regression
        ret['estimator'] = self.estimator
        ret['encoder'] = self.encoder
        if deep:
            ret['cells'] = copy.deepcopy(self.cells)
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
        if 'feature_slices' in params:
            self.feature_slices = params['feature_slices']
        if 'train_only_features_to_minimize' in params:
            self.train_only_features_to_minimize = params['train_only_features_to_minimize']
        if 'is_regression' in params:
            self.is_regression = params['is_regression']
        if 'cells' in params:
            self.cells = params['cells']
        if 'estimator' in params:
            self.estimator = params['estimator']
        if 'encoder' in params:
            self.encoder = params['encoder']
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
        Return the last calculated NCP scores. NCP score is calculated upon calling `fit` (on the training data),
        `transform' (on the test data) or when explicitly calling `calculate_ncp` and providing it a dataset.

        :return: NCPScores object, that contains a score corresponding to the last fit call, one for the last
        transform call, and a score based on global generalizations.
        """
        return self._ncp_scores

    def fit_transform(self, X: Optional[DATA_PANDAS_NUMPY_TYPE] = None, y: Optional[DATA_PANDAS_NUMPY_TYPE] = None,
                      features_names: Optional[list] = None, dataset: Optional[ArrayDataset] = None):
        """
        Learns the generalizations based on training data, and applies them to the data. Also sets the fit_score,
        transform_score and generalizations_score in self.ncp.

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
        if not self.generalize_using_transform:
            raise ValueError('fit_transform method called even though generalize_using_transform parameter was False. '
                             'This can lead to inconsistent results.')
        self.fit(X, y, features_names, dataset=dataset)
        return self.transform(X, features_names, dataset=dataset)

    def fit(self, X: Optional[DATA_PANDAS_NUMPY_TYPE] = None, y: Optional[DATA_PANDAS_NUMPY_TYPE] = None,
            features_names: Optional = None, dataset: ArrayDataset = None):
        """Learns the generalizations based on training data. Also sets the fit_score and generalizations_score in
        self.ncp.

        :param X: The training input samples.
        :type X: {array-like, sparse matrix}, shape (n_samples, n_features), optional
        :param y: The target values. This should contain the predictions of the original model on ``X``.
        :type y: array-like, shape (n_samples,), optional
        :param features_names: The feature names, in the order that they appear in the data. Should be provided when
                               passing the data as ``X`` as a numpy array
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
            if self.feature_slices:
                temp_list = []
                for slice in self.feature_slices:
                    new_slice = [str(i) for i in slice]
                    if not all(elem in self._features for elem in new_slice):
                        raise ValueError('features in slices should be a subset of features names')
                    temp_list.append(new_slice)
                self.feature_slices = temp_list
            x_qi = x.loc[:, self.features_to_minimize]

            # divide dataset into train and test
            used_data = x
            if self.train_only_features_to_minimize:
                used_data = x_qi
            if self.is_regression:
                x_train, x_test, y_train, y_test = train_test_split(x, dataset.get_labels(), test_size=0.4,
                                                                    random_state=14)
            else:
                try:
                    x_train, x_test, y_train, y_test = train_test_split(x, dataset.get_labels(),
                                                                        stratify=dataset.get_labels(), test_size=0.4,
                                                                        random_state=18)
                except ValueError:
                    print('Could not stratify split due to uncommon class value, doing unstratified split instead')
                    x_train, x_test, y_train, y_test = train_test_split(x, dataset.get_labels(), test_size=0.4,
                                                                        random_state=18)

            x_train_qi = x_train.loc[:, self.features_to_minimize]
            x_test_qi = x_test.loc[:, self.features_to_minimize]
            used_x_train = x_train
            used_x_test = x_test
            if self.train_only_features_to_minimize:
                used_x_train = x_train_qi
                used_x_test = x_test_qi

            # collect feature data (such as min, max)
            self._feature_data = self._get_feature_data(x)

            self.cells = []
            self._categorical_values = {}

            if self.is_regression:
                self._dt = DecisionTreeRegressor(random_state=10, min_samples_split=2, min_samples_leaf=1)
            else:
                self._dt = DecisionTreeClassifier(random_state=0, min_samples_split=2,
                                                  min_samples_leaf=1)

            # prepare data for DT
            self._encode_categorical_features(used_data, save_mapping=True)
            x_prepared = self._encode_categorical_features(used_x_train)
            self._dt.fit(x_prepared, y_train)
            x_prepared_test = self._encode_categorical_features(used_x_test)

            self._calculate_cells()
            self._modify_cells()
            # features that are not from QI should not be part of generalizations
            for feature in self._features:
                if feature not in self.features_to_minimize:
                    self._remove_feature_from_cells(self.cells, self._cells_by_id, feature)

            nodes = self._get_nodes_level(0)
            self._attach_cells_representatives(x_prepared, used_x_train, y_train, nodes)

            # self._cells currently holds the generalization created from the tree leaves
            generalized = self._generalize(x_test, x_prepared_test, nodes)

            # check accuracy
            accuracy = self._calculate_accuracy(generalized, y_test, self.estimator, self.encoder)
            print('Initial accuracy of model on generalized data, relative to original model predictions '
                  '(base generalization derived from tree, before improvements): %f' % accuracy)

            # if accuracy above threshold, improve generalization
            if accuracy > self.target_accuracy:
                print('Improving generalizations')
                self._level = 0
                while accuracy > self.target_accuracy:
                    self._level += 1
                    cells_previous_iter = self.cells
                    generalization_prev_iter = self._generalizations
                    cells_by_id_prev = self._cells_by_id
                    nodes = self._get_nodes_level(self._level)

                    try:
                        self._calculate_level_cells(self._level)
                    except TypeError as e:
                        print(e)
                        self._level -= 1
                        break

                    self._attach_cells_representatives(x_prepared, used_x_train, y_train, nodes)

                    generalized = self._generalize(x_test, x_prepared_test, nodes)
                    accuracy = self._calculate_accuracy(generalized, y_test, self.estimator, self.encoder)
                    # if accuracy passed threshold roll back to previous iteration generalizations
                    if accuracy < self.target_accuracy:
                        self.cells = cells_previous_iter
                        self._generalizations = generalization_prev_iter
                        self._cells_by_id = cells_by_id_prev
                        self._level -= 1
                        break
                    else:
                        print('Pruned tree to level: %d, new relative accuracy: %f' % (self._level, accuracy))

            # if accuracy below threshold, improve accuracy by removing features from generalization
            elif accuracy < self.target_accuracy:
                print('Improving accuracy')
                while accuracy < self.target_accuracy:
                    removed_feature = self._remove_feature_from_generalization(x_test, x_prepared_test,
                                                                               nodes, y_test,
                                                                               self._feature_data, accuracy,
                                                                               self.generalize_using_transform)
                    if removed_feature is None:
                        break

                    generalized = self._generalize(x_test, x_prepared_test, nodes)
                    accuracy = self._calculate_accuracy(generalized, y_test, self.estimator, self.encoder)
                    print('Removed feature: %s, new relative accuracy: %f' % (removed_feature, accuracy))

            # self._cells currently holds the chosen generalization based on target accuracy

            # calculate iLoss
            x_test_dataset = ArrayDataset(x_test, features_names=self._features)
            self._ncp_scores.fit_score = self.calculate_ncp(x_test_dataset)
            self._ncp_scores.generalizations_score = self.calculate_ncp(x_test_dataset)
        else:
            print('No fitting was performed as some information was missing')
            if not self.estimator:
                print('No estimator provided')
            elif not dataset:
                print('No data provided')
            elif dataset.get_samples() is None:
                print('No samples provided')
            elif dataset.get_labels() is None:
                print('No labels provided')

        # Return the transformer
        return self

    def transform(self, X: Optional[DATA_PANDAS_NUMPY_TYPE] = None, features_names: Optional[list] = None,
                  dataset: Optional[ArrayDataset] = None):
        """ Transforms data records to representative points. Also sets the transform_score in self.ncp.

        :param X: The training input samples.
        :type X: {array-like, sparse matrix}, shape (n_samples, n_features), optional
        :param features_names: The feature names, in the order that they appear in the data. Should be provided when
                               passing the data as ``X`` as a numpy array
        :type features_names: list of strings, optional
        :param dataset: Data wrapper containing the training input samples and the predictions of the original model
                        on the training data. Either ``X`` OR ``dataset`` need to be provided, not both.
        :type dataset: `ArrayDataset`, optional
        :return: Array containing the representative values to which each record in ``X`` is mapped, as numpy array or
                 pandas DataFrame (depending on the type of ``X``), shape (n_samples, n_features)
        """
        if not self.generalize_using_transform:
            raise ValueError('transform method called even though generalize_using_transform parameter was False. This '
                             'can lead to inconsistent results.')
        transformed = self._inner_transform(X, features_names, dataset)
        transformed_dataset = ArrayDataset(transformed, features_names=self._features)
        self._ncp_scores.transform_score = self.calculate_ncp(transformed_dataset)
        return transformed

    def calculate_ncp(self, samples: ArrayDataset):
        """
        Compute the NCP score of the generalization. Calculation is based on the value of the
        generalize_using_transform param. If samples are provided, updates stored ncp value to the one computed on the
        provided data. If samples not provided, returns the last NCP score computed by the `fit` or `transform` method.

        Based on the NCP score presented in: Ghinita, G., Karras, P., Kalnis, P., Mamoulis, N.: Fast data anonymization
        with low information loss (https://www.vldb.org/conf/2007/papers/research/p758-ghinita.pdf)

        :param samples: The input samples to compute the NCP score on.
        :type samples: ArrayDataset, optional. feature_names should be set.
        :return: NCP score as float.
        """
        if not samples.features_names:
            raise ValueError('features_names should be set in input ArrayDataset.')
        samples_pd = pd.DataFrame(samples.get_samples(), columns=samples.features_names)
        if self._features is None:
            self._features = samples.features_names
        if self._feature_data is None:
            self._feature_data = self._get_feature_data(samples_pd)
        total_samples = samples_pd.shape[0]

        if self.generalize_using_transform:
            generalizations = self._calculate_cell_generalizations()
            # count how many records are mapped to each cell
            counted = np.zeros(samples_pd.shape[0])  # to mark records we already counted
            ncp = 0
            for cell in self.cells:
                count = self._get_record_count_for_cell(samples_pd, cell, counted)
                range_counts = {}
                category_counts = {}
                for feature in cell['ranges']:
                    range_counts[feature] = [count]
                for feature in cell['categories']:
                    category_counts[feature] = [count]
                ncp += self._calc_ncp_for_generalization(generalizations[cell['id']], range_counts, category_counts,
                                                         total_samples)
        else:  # use generalizations
            generalizations = self.generalizations
            range_counts = self._find_range_counts(samples_pd, generalizations['ranges'])
            category_counts = self._find_category_counts(samples_pd, generalizations['categories'])
            ncp = self._calc_ncp_for_generalization(generalizations, range_counts, category_counts, total_samples)

        return ncp

    def _inner_transform(self, x: Optional[DATA_PANDAS_NUMPY_TYPE] = None, features_names: Optional[list] = None,
                         dataset: Optional[ArrayDataset] = None):
        # Check if fit has been called
        msg = 'This %(name)s instance is not initialized yet. ' \
              'Call ‘fit’ or ‘set_params’ with ' \
              'appropriate arguments before using this method.'
        check_is_fitted(self, ['cells'], msg=msg)

        if x is not None:
            if dataset is not None:
                raise ValueError('Either x OR dataset need to be provided, not both')
            else:
                dataset = ArrayDataset(x, features_names=features_names)
        elif dataset is None:
            raise ValueError('Either x OR dataset need to be provided, not both')
        if dataset and dataset.features_names:
            if self._features is None:
                self._features = dataset.features_names
        if dataset and dataset.get_samples() is not None:
            x_pd = pd.DataFrame(dataset.get_samples(), columns=self._features)

        if x_pd.shape[1] != self._n_features and self._n_features != 0:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')

        if not self._features:
            self._features = [i for i in range(x_pd.shape[1])]

        if self._dt:  # only works if fit was called previously (but much more efficient)
            nodes = self._get_nodes_level(self._level)
            QI = x_pd.loc[:, self.features_to_minimize]
            used_x = x_pd
            if self.train_only_features_to_minimize:
                used_x = QI
            prepared = self._encode_categorical_features(used_x)
            generalized = self._generalize_from_tree(x_pd, prepared, nodes, self.cells, self._cells_by_id)
        else:
            mapped = np.zeros(x_pd.shape[0])  # to mark records we already mapped
            all_indexes = []
            for cell in self.cells:
                indexes = self._get_record_indexes_for_cell(x_pd, cell, mapped)
                all_indexes.append(indexes)
            generalized = self._generalize_indexes(x_pd, self.cells, all_indexes)

        if dataset and dataset.is_pandas:
            return generalized
        elif isinstance(x, pd.DataFrame):
            return generalized
        return generalized.to_numpy()

    def _calc_ncp_for_generalization(self, generalization, range_counts, category_counts, total_count):
        total_ncp = 0
        total_features = len(generalization['untouched'])
        ranges = generalization['ranges']
        categories = generalization['categories']

        # suppressed features are already taken care of within _calc_ncp_numeric
        for feature in ranges.keys():
            feature_ncp = self._calc_ncp_numeric(ranges[feature], range_counts[feature],
                                                 self._feature_data[feature], total_count)
            total_ncp = total_ncp + feature_ncp
            total_features += 1
        for feature in categories.keys():
            feature_ncp = self._calc_ncp_categorical(categories[feature], category_counts[feature],
                                                     self._feature_data[feature],
                                                     total_count)
            total_ncp = total_ncp + feature_ncp
            total_features += 1
        if total_features == 0:
            return 0
        return total_ncp / total_features

    @staticmethod
    def _calc_ncp_categorical(categories, category_count, feature_data, total):
        category_sizes = [len(g) if len(g) > 1 else 0 for g in categories]
        normalized_category_sizes = [s * n / total for s, n in zip(category_sizes, category_count)]
        average_group_size = sum(normalized_category_sizes) / len(normalized_category_sizes)
        return average_group_size / feature_data['range']  # number of values in category

    @staticmethod
    def _calc_ncp_numeric(range, range_count, feature_data, total):
        # if there are no ranges, feature is suppressed and iLoss is 1
        if not range:
            return 1
        # range only contains the split values, need to add min and max value of feature
        # to enable computing sizes of all ranges
        new_range = [feature_data['min']] + range + [feature_data['max']]
        range_sizes = [b - a for a, b in zip(new_range[::1], new_range[1::1])]
        normalized_range_sizes = [s * n / total for s, n in zip(range_sizes, range_count)]
        average_range_size = sum(normalized_range_sizes) / len(normalized_range_sizes)
        return average_range_size / (feature_data['max'] - feature_data['min'])

    def _get_feature_data(self, x):
        feature_data = {}
        for feature in self._features:
            if feature not in feature_data.keys():
                fd = {}
                values = list(x.loc[:, feature])
                if feature not in self.categorical_features and feature not in self.all_one_hot_features:
                    fd['min'] = min(values)
                    fd['max'] = max(values)
                    fd['range'] = max(values) - min(values)
                else:
                    fd['range'] = len(np.unique(values))
                feature_data[feature] = fd
        return feature_data

    def _get_record_indexes_for_cell(self, x, cell, mapped):
        indexes = []
        for index, row in x.iterrows():
            if not mapped.item(index) and self._cell_contains(cell, row, index, mapped):
                indexes.append(index)
        return indexes

    def _get_record_count_for_cell(self, x, cell, mapped):
        count = 0
        for index, (_, row) in enumerate(x.iterrows()):
            if not mapped.item(index) and self._cell_contains(cell, row, index, mapped):
                count += 1
        return count

    def _cell_contains(self, cell, row, index, mapped):
        for i, feature in enumerate(self._features):
            if feature in cell['ranges']:
                if not self._cell_contains_numeric(i, cell['ranges'][feature], row):
                    return False
            elif feature in cell['categories']:
                if not self._cell_contains_categorical(i, cell['categories'][feature], row):
                    return False
            elif feature in cell['untouched']:
                continue
            else:
                raise TypeError("feature " + str(feature) + " not found in cell " + str(cell['id']))
        # Mark as mapped
        mapped.itemset(index, 1)
        return True

    def _encode_categorical_features(self, x, save_mapping=False):
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
                    all_values = x.loc[:, feature]
                    values = list(all_values.unique())
                    if save_mapping:
                        self._categorical_values[feature] = values
                    x[feature] = pd.Categorical(x.loc[:, feature], categories=self._categorical_values[feature],
                                                ordered=False)
                    ohe = pd.get_dummies(x[feature], prefix=feature)
                    if save_mapping:
                        for one_hot_vector_feature in ohe.columns:
                            self._one_hot_vector_features_to_features[one_hot_vector_feature] = feature
                    x = pd.concat([x, ohe], axis=1)
                    features_to_remove.append(feature)
                except KeyError:
                    print("feature " + feature + "not found in training data")

        new_data = x.drop(features_to_remove, axis=1)
        if save_mapping:
            self._encoded_features = new_data.columns
        return new_data

    @staticmethod
    def _cell_contains_numeric(index, range, row):
        # convert row to ndarray to allow indexing
        a = np.array(row)
        value = a.item(index)
        if range['start']:
            if value <= range['start']:
                return False
        if range['end']:
            if value > range['end']:
                return False
        return True

    @staticmethod
    def _cell_contains_categorical(index, range, row):
        # convert row to ndarray to allow indexing
        a = np.array(row)
        value = a.item(index)
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
            hist = self._dt.tree_.value[node]
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
        label_hist = self._dt.tree_.value[node]
        if isinstance(self._dt.classes_, list):
            return [self._dt.classes_[output][class_index]
                    for output, class_index in enumerate(np.argmax(label_hist, axis=1))]
        return [self._dt.classes_[np.argmax(label_hist[0])]]

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
                # features that were already one-hot encoded. Legal values should be 0 or 1
                elif feature in self.all_one_hot_features:
                    if feature not in new_cell['categories'].keys():
                        new_cell['categories'][feature] = []
                    if feature in cell['ranges']:
                        range = cell['ranges'][feature]
                        if range['start'] is None and range['end'] < 1:
                            feature_value = 0
                        elif range['end'] is None and range['start'] > 0:
                            feature_value = 1
                        else:
                            raise ValueError('Illegal range for 1-hot encoded feature')
                        new_cell['categories'][feature] = [feature_value]

                        # need to add other columns that represent same 1-hot encoded feature

                        # search for feature group:
                        other_features, encoded = self._get_other_features_in_encoding(feature, self.feature_slices)
                        for other_feature in other_features:
                            if feature_value == 1:
                                new_cell['categories'][other_feature] = [0]
                            elif len(encoded) == 2:
                                new_cell['categories'][other_feature] = [1]
                            elif (other_feature not in new_cell['categories'].keys()
                                  or len(new_cell['categories'][other_feature]) == 0):
                                new_cell['categories'][other_feature] = [0, 1]
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
        new_cell['hist'] = left_cell['hist'] + right_cell['hist']
        if isinstance(self._dt.classes_, list):
            new_cell['label'] = [self._dt.classes_[output][class_index]
                                 for output, class_index in enumerate(np.argmax(new_cell['hist'], axis=1))]
        else:
            new_cell['label'] = [self._dt.classes_[np.argmax(new_cell['hist'][0])]]

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

    def _attach_cells_representatives(self, prepared_data, original_train_features, label_feature, level_nodes):
        # prepared data include one hot encoded categorical data,
        # if there is no categorical data prepared data is original data
        nodeIds = self._find_sample_nodes(prepared_data, level_nodes)
        for cell in self.cells:
            cell['representative'] = {}
            # get all rows in cell
            indexes = [i for i, x in enumerate(nodeIds) if x == cell['id']]
            original_rows = original_train_features.iloc[indexes]
            sample_rows = prepared_data.iloc[indexes]

            # get rows with matching label
            if self.is_regression or (len(label_feature.shape) > 1 and label_feature.shape[1] > 1):
                match_samples = sample_rows
                match_rows = original_rows
            else:
                labels_df = pd.DataFrame(label_feature, columns=['label'])
                sample_labels = labels_df.iloc[indexes]['label'].values.tolist()
                indexes = [i for i, label in enumerate(sample_labels) if label == cell['label'][0]]
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
            # since this is an actual row from the data, correct one-hot encoding is already guaranteed
            row = match_rows.iloc[min]
            for feature in cell['ranges'].keys():
                cell['representative'][feature] = row[feature]
            for feature in cell['categories'].keys():
                cell['representative'][feature] = row[feature]

    def _find_sample_nodes(self, samples, nodes):
        paths = self._dt.decision_path(samples).toarray()
        nodeSet = set(nodes)
        return [(list(set([i for i, v in enumerate(p) if v == 1]) & nodeSet))[0] for p in paths]

    # method for applying generalizations (for global generalization-based acuuracy) without dt
    def _generalize_from_generalizations(self, original_data, generalizations):
        sample_indexes = self._map_to_ranges_categories(original_data,
                                                        generalizations['ranges'],
                                                        generalizations['categories'])
        original_data_generalized = pd.DataFrame(original_data, columns=self._features, copy=True)
        for feature in self._generalizations['categories']:
            if 'untouched' not in generalizations or feature not in generalizations['untouched']:
                for g_index, group in enumerate(generalizations['categories'][feature]):
                    indexes = [i for i, s in enumerate(sample_indexes) if s[feature] == g_index]
                    if indexes:
                        rows = original_data_generalized.iloc[indexes]
                        rows[feature] = generalizations['category_representatives'][feature][g_index]
        for feature in self._generalizations['ranges']:
            if 'untouched' not in generalizations or feature not in generalizations['untouched']:
                for r_index, range in enumerate(generalizations['ranges'][feature]):
                    indexes = [i for i, s in enumerate(sample_indexes) if s[feature] == r_index]
                    if indexes:
                        rows = original_data_generalized.iloc[indexes]
                        rows[feature] = generalizations['range_representatives'][feature][r_index]
        return original_data_generalized

    def _generalize_from_tree(self, original_data, prepared_data, level_nodes, cells, cells_by_id):
        mapping_to_cells = self._map_to_cells(prepared_data, level_nodes, cells_by_id)
        all_indexes = []
        for i in range(len(cells)):
            # get the indexes of all records that map to this cell
            indexes = [j for j in mapping_to_cells if mapping_to_cells[j]['id'] == cells[i]['id']]
            all_indexes.append(indexes)
        return self._generalize_indexes(original_data, cells, all_indexes)

    def _generalize_indexes(self, original_data, cells, all_indexes):
        # prepared data include one hot encoded categorical data + QI
        dtypes = original_data.dtypes.to_dict()
        new_dtypes = {}
        for t in dtypes.keys():
            new_dtypes[t] = pd.Series(dtype=dtypes[t].name)
            dtypes[t] = dtypes[t].name
        representatives = pd.DataFrame(new_dtypes)  # empty except for columns
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

        original_data_generalized = original_data_generalized.astype(dtype=dtypes)
        return original_data_generalized

    def _generalize(self, data, data_prepared, nodes):
        self._calculate_generalizations(data)
        if self.generalize_using_transform:
            generalized = self._generalize_from_tree(data, data_prepared, nodes, self.cells,
                                                     self._cells_by_id)
        else:
            generalized = self._generalize_from_generalizations(data, self.generalizations)
        return generalized

    @staticmethod
    def _map_to_ranges_categories(samples, ranges, categories):
        all_sample_indexes = []
        for _, row in samples.iterrows():
            sample_indexes = {}
            for feature in ranges:
                if not ranges[feature]:
                    # no values means whole range
                    sample_indexes[feature] = 0
                else:
                    for index, value in enumerate(ranges[feature]):
                        if row[feature] <= value:
                            sample_indexes[feature] = index
                            break
                    sample_indexes[feature] = index + 1
            for feature in categories:
                for g_index, group in enumerate(categories[feature]):
                    if row[feature] in group:
                        sample_indexes[feature] = g_index
                        break
            all_sample_indexes.append(sample_indexes)
        return all_sample_indexes

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
                                            current_accuracy, generalize_using_transform):
        # prepared data include one hot encoded categorical data,
        # if there is no categorical data prepared data is original data
        feature = self._get_feature_to_remove(original_data, prepared_data, nodes, labels, feature_data,
                                              current_accuracy, generalize_using_transform)
        if feature is None:
            return None
        self._remove_feature_from_cells(self.cells, self._cells_by_id, feature)
        return feature

    def _get_feature_to_remove(self, original_data, prepared_data, nodes, labels, feature_data, current_accuracy,
                               generalize_using_transform):
        # prepared data include one hot encoded categorical data,
        # if there is no categorical data prepared data is original data
        # We want to remove features with low iLoss (NCP) and high accuracy gain
        # (after removing them)
        ranges = self._generalizations['ranges']
        range_counts = self._find_range_counts(original_data, ranges)
        total = prepared_data.size
        range_min = sys.float_info.max
        remove_feature = None
        categories = self.generalizations['categories']
        category_counts = self._find_category_counts(original_data, categories)

        for feature in ranges.keys():
            if feature not in self._generalizations['untouched']:
                if generalize_using_transform:
                    feature_ncp = self._calculate_ncp_for_feature_from_cells(feature, feature_data, original_data)
                else:
                    feature_ncp = self._calc_ncp_numeric(ranges[feature],
                                                         range_counts[feature],
                                                         feature_data[feature],
                                                         total)
                if feature_ncp > 0:
                    feature_ncp = self._normalize_ncp_by_accuracy_gain(original_data, prepared_data, nodes, feature,
                                                                       feature_ncp, labels, current_accuracy)

                if feature_ncp < range_min:
                    range_min = feature_ncp
                    remove_feature = feature

        for feature in categories.keys():
            if feature not in self.generalizations['untouched']:
                if generalize_using_transform:
                    feature_ncp = self._calculate_ncp_for_feature_from_cells(feature, feature_data, original_data)
                else:
                    feature_ncp = self._calc_ncp_categorical(categories[feature],
                                                             category_counts[feature],
                                                             feature_data[feature],
                                                             total)
                if feature_ncp > 0:
                    feature_ncp = self._normalize_ncp_by_accuracy_gain(original_data, prepared_data, nodes, feature,
                                                                       feature_ncp, labels, current_accuracy)

                if feature_ncp < range_min:
                    range_min = feature_ncp
                    remove_feature = feature

        print('feature to remove: ' + (str(remove_feature) if remove_feature is not None else 'none'))
        return remove_feature

    def _calculate_ncp_for_feature_from_cells(self, feature, feature_data, samples_pd):
        # count how many records are mapped to each cell
        counted = np.zeros(samples_pd.shape[0])  # to mark records we already counted
        total = samples_pd.shape[0]
        feature_ncp = 0
        for cell in self.cells:
            count = self._get_record_count_for_cell(samples_pd, cell, counted)
            generalizations = self._calculate_generalizations_for_cell(cell)
            cell_ncp = 0
            if feature in cell['ranges']:
                cell_ncp = self._calc_ncp_numeric(generalizations['ranges'][feature],
                                                  [count],
                                                  feature_data[feature],
                                                  total)
            elif feature in cell['categories']:
                cell_ncp = self._calc_ncp_categorical(generalizations['categories'][feature],
                                                      [count],
                                                      feature_data[feature],
                                                      total)
            feature_ncp += cell_ncp
        return feature_ncp

    def _normalize_ncp_by_accuracy_gain(self, original_data, prepared_data, nodes, feature, feature_ncp, labels,
                                        current_accuracy):
        new_cells = copy.deepcopy(self.cells)
        cells_by_id = copy.deepcopy(self._cells_by_id)
        self._remove_feature_from_cells(new_cells, cells_by_id, feature)
        generalized = self._generalize_from_tree(original_data, prepared_data, nodes, new_cells,
                                                 cells_by_id)
        accuracy = self._calculate_accuracy(generalized, labels, self.estimator, self.encoder)
        accuracy_gain = accuracy - current_accuracy
        if accuracy_gain < 0:
            accuracy_gain = 0
        if accuracy_gain != 0:
            feature_ncp = feature_ncp / accuracy_gain
        return feature_ncp

    def _calculate_generalizations(self, samples: Optional[pd.DataFrame] = None):
        ranges, range_representatives = self._calculate_ranges(self.cells)
        categories, category_representatives = self._calculate_categories(self.cells)
        self._generalizations = {'ranges': ranges,
                                 'categories': categories,
                                 'untouched': self._calculate_untouched(self.cells)}
        self._remove_categorical_untouched(self._generalizations)
        # compute representative value for each feature (based on data)
        if samples is not None:
            sample_indexes = self._map_to_ranges_categories(samples,
                                                            self._generalizations['ranges'],
                                                            self._generalizations['categories'])
            # categorical - use most common value
            old_category_representatives = category_representatives
            category_representatives = {}
            done = set()
            for feature in self._generalizations['categories']:
                if feature not in done:
                    category_representatives[feature] = []
                    for g_index, group in enumerate(self._generalizations['categories'][feature]):
                        indexes = [i for i, s in enumerate(sample_indexes) if s[feature] == g_index]
                        if indexes:
                            rows = samples.iloc[indexes]
                            if feature in self.all_one_hot_features:
                                other_features, encoded = self._get_other_features_in_encoding(feature,
                                                                                               self.feature_slices)
                                values = rows.loc[:, encoded].to_numpy()
                                unique_rows, counts = np.unique(values, axis=0, return_counts=True)
                                rep = unique_rows[np.argmax(counts)]
                                for i, e in enumerate(encoded):
                                    done.add(e)
                                    if e not in category_representatives.keys():
                                        category_representatives[e] = []
                                    category_representatives[e].append(rep[i])
                            else:
                                values = rows[feature]
                                category = Counter(values).most_common(1)[0][0]
                                category_representatives[feature].append(category)
                        else:
                            category_representatives[feature].append(old_category_representatives[feature][g_index])

            # numerical - use actual value closest to mean
            old_range_representatives = range_representatives
            range_representatives = {}
            for feature in self._generalizations['ranges']:
                range_representatives[feature] = []
                # find the mean value (per feature)
                for index in range(len(self._generalizations['ranges'][feature])):
                    indexes = [i for i, s in enumerate(sample_indexes) if s[feature] == index]
                    if indexes:
                        rows = samples.iloc[indexes]
                        values = rows[feature]
                        median = np.median(values)
                        min_value = max(values)
                        min_dist = float("inf")
                        for value in values:
                            # euclidean distance between two floating point values
                            dist = abs(value - median)
                            if dist < min_dist:
                                min_dist = dist
                                min_value = value
                        range_representatives[feature].append(min_value)
                    else:
                        range_representatives[feature].append(old_range_representatives[feature][index])
        self._generalizations['category_representatives'] = category_representatives
        self._generalizations['range_representatives'] = range_representatives

    def _calculate_generalizations_for_cell(self, cell):
        ranges, range_representatives = self._calculate_ranges([cell])
        categories, category_representatives = self._calculate_categories([cell])
        generalizations = {'ranges': ranges,
                           'categories': categories,
                           'untouched': self._calculate_untouched([cell]),
                           'range_representatives': range_representatives,
                           'category_representatives': category_representatives}
        self._remove_categorical_untouched(generalizations)
        return generalizations

    def _calculate_cell_generalizations(self):
        # calculate generalizations separately per cell
        cell_generalizations = {}
        for cell in self.cells:
            cell_generalizations[cell['id']] = self._calculate_generalizations_for_cell(cell)
        return cell_generalizations

    @staticmethod
    def _find_range_counts(samples, ranges):
        range_counts = {}
        last_value = None
        for r in ranges.keys():
            range_counts[r] = []
            # if empty list, all samples should be counted
            if not ranges[r]:
                range_counts[r].append(samples.shape[0])
            else:
                for value in ranges[r]:
                    counter = [item for item in samples[r] if int(item) <= value]
                    range_counts[r].append(len(counter))
                    last_value = value
                counter = [item for item in samples[r] if int(item) > last_value]
                range_counts[r].append(len(counter))
        return range_counts

    @staticmethod
    def _find_category_counts(samples, categories):
        category_counts = {}
        for c in categories.keys():
            category_counts[c] = []
            for value in categories[c]:
                category_counts[c].append(len(samples.loc[samples[c].isin(value)]))
        return category_counts

    @staticmethod
    def _calculate_ranges(cells):
        ranges = {}
        range_representatives = {}
        for cell in cells:
            for feature in [key for key in cell['ranges'].keys() if
                            'untouched' not in cell or key not in cell['untouched']]:
                if feature not in ranges.keys():
                    ranges[feature] = []
                if cell['ranges'][feature]['start'] is not None:
                    ranges[feature].append(cell['ranges'][feature]['start'])
                if cell['ranges'][feature]['end'] is not None:
                    ranges[feature].append(cell['ranges'][feature]['end'])
        # default representative values (computed with no data)
        for feature in ranges.keys():
            range_representatives[feature] = []
            if not ranges[feature]:
                # no values means the complete range. Without data we cannot know what to put here.
                # Using 0 as a placeholder.
                range_representatives[feature].append(0)
            else:
                ranges[feature] = list(set(ranges[feature]))
                ranges[feature].sort()
                prev_value = 0
                for index, value in enumerate(ranges[feature]):
                    if index == 0:
                        # for first range, use min value
                        range_representatives[feature].append(value)
                    else:
                        # use middle of range (this will be a float)
                        range_representatives[feature].append((value - prev_value) / 2)
                    prev_value = value
                # for last range use max value + 1
                range_representatives[feature].append(prev_value + 1)
        return ranges, range_representatives

    def _calculate_categories(self, cells):
        categories = {}
        category_representatives = {}
        categorical_features_values = GeneralizeToRepresentative._calculate_categorical_features_values(cells)
        assigned_features = set()
        for feature in categorical_features_values.keys():
            partitions = []
            category_representatives[feature] = []
            values = categorical_features_values[feature]
            assigned_values = set()
            for i in range(len(values)):
                value1 = values[i]
                if value1 in assigned_values:
                    continue
                partition = [value1]
                assigned_values.add(value1)
                for j in range(len(values)):
                    if j <= i:
                        continue
                    value2 = values[j]
                    if GeneralizeToRepresentative._are_inseparable(cells, feature, value1, value2):
                        partition.append(value2)
                        assigned_values.add(value2)
                partitions.append(partition)
                # default representative values (computed with no data)
                # for 1-hot encoded features, the first encountered feature will get the value 1 and the rest 0
                if len(partition) > 1 and feature in self.all_one_hot_features:
                    other_features, _ = self._get_other_features_in_encoding(feature, self.feature_slices)
                    assigned = False
                    for other_feature in other_features:
                        if other_feature in assigned_features:
                            category_representatives[feature].append(0)
                            assigned = True
                            break
                    if not assigned:
                        category_representatives[feature].append(1)
                    assigned_features.add(feature)
                else:
                    category_representatives[feature].append(partition[0])  # random
            categories[feature] = partitions
        return categories, category_representatives

    @staticmethod
    def _get_other_features_in_encoding(feature, feature_slices):
        for encoded in feature_slices:
            if feature in encoded:
                return (list(set(encoded) - {feature})), encoded
        return [], []

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

    def _remove_feature_from_cells(self, cells, cells_by_id, feature):
        if feature in self.all_one_hot_features:
            for encoded in self.feature_slices:
                if feature in encoded:
                    self._remove_feature_from_cells_internal(cells, cells_by_id, encoded)
        else:
            self._remove_feature_from_cells_internal(cells, cells_by_id, [feature])

    @staticmethod
    def _remove_feature_from_cells_internal(cells, cells_by_id, features):
        for cell in cells:
            if 'untouched' not in cell:
                cell['untouched'] = []
            for feature in features:
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

    @staticmethod
    def _calculate_accuracy(generalized, y_test, estimator, encoder):
        generalized_data = encoder.transform(generalized) if encoder else generalized
        return estimator.score(ArrayDataset(generalized_data, y_test))
