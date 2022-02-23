import numpy as np
import pandas as pd
from scipy.spatial import distance
from collections import Counter

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder

from typing import Union, Optional


class Anonymize:
    """
    Class for performing tailored, model-guided anonymization of training datasets for ML models.

    Based on the implementation described in: https://arxiv.org/abs/2007.13086

    Parameters
    ----------
    k : int
        The privacy parameter that determines the number of records that will be indistinguishable from each
        other (when looking at the quasi identifiers). Should be at least 2.
    quasi_identifiers : np.ndarray or list
        The features that need to be minimized in case of pandas data, and indexes of features
        in case of numpy data.
    categorical_features : list, optional
        The list of categorical features (should only be supplied when passing data as a
        pandas dataframe.
    is_regression : Bool, optional
        Whether the model is a regression model or not (if False, assumes
        a classification model). Default is False.
    """

    def __init__(self, k: int, quasi_identifiers: Union[np.ndarray, list], categorical_features: Optional[list] = None,
                 is_regression=False):
        if k < 2:
            raise ValueError("k should be a positive integer with a value of 2 or higher")
        if quasi_identifiers is None or len(quasi_identifiers) < 1:
            raise ValueError("The list of quasi-identifiers cannot be empty")

        self.k = k
        self.quasi_identifiers = quasi_identifiers
        self.categorical_features = categorical_features
        self.is_regression = is_regression

    def anonymize(self, x: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.DataFrame]) \
            -> Union[np.ndarray, pd.DataFrame]:
        """
        Method for performing model-guided anonymization.

        :param x: The training data for the model. If provided as a pandas dataframe, may contain both numeric and
                  categorical data.
        :param y: The predictions of the original model on the training data.
        :return: An array containing the anonymized training dataset.
        """
        if type(x) == np.ndarray:
            return self._anonymize_ndarray(x.copy(), y)
        else:  # pandas
            if not self.categorical_features:
                raise ValueError('When supplying a pandas dataframe, categorical_features must be defined')
            return self._anonymize_pandas(x.copy(), y)

    def _anonymize_ndarray(self, x, y):
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y should have same number of rows")
        x_anonymizer_train = x[:, self.quasi_identifiers]
        if x.dtype.kind not in 'iufc':
            x_prepared = self._modify_categorical_features(x_anonymizer_train)
        else:
            x_prepared = x_anonymizer_train
        if self.is_regression:
            self.anonymizer = DecisionTreeRegressor(random_state=10, min_samples_split=2, min_samples_leaf=self.k)
        else:
            self.anonymizer = DecisionTreeClassifier(random_state=10, min_samples_split=2, min_samples_leaf=self.k)
        self.anonymizer.fit(x_prepared, y)
        cells_by_id = self._calculate_cells(x, x_prepared)
        return self._anonymize_data_numpy(x, x_prepared, cells_by_id)

    def _anonymize_pandas(self, x, y):
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y should have same number of rows")
        x_anonymizer_train = x.loc[:, self.quasi_identifiers]
        # need to one-hot encode before training the decision tree
        x_prepared = self._modify_categorical_features(x_anonymizer_train)
        if self.is_regression:
            self.anonymizer = DecisionTreeRegressor(random_state=10, min_samples_split=2, min_samples_leaf=self.k)
        else:
            self.anonymizer = DecisionTreeClassifier(random_state=10, min_samples_split=2, min_samples_leaf=self.k)
        self.anonymizer.fit(x_prepared, y)
        cells_by_id = self._calculate_cells(x, x_prepared)
        return self._anonymize_data_pandas(x, x_prepared, cells_by_id)

    def _calculate_cells(self, x, x_anonymizer_train):
        # x is original data, x_anonymizer_train is only QIs + 1-hot encoded
        cells_by_id = {}
        leaves = []
        for node, feature in enumerate(self.anonymizer.tree_.feature):
            if feature == -2:  # leaf node
                leaves.append(node)
                hist = [int(i) for i in self.anonymizer.tree_.value[node][0]]
                # TODO we may change the method for choosing representative for cell
                # label_hist = self.anonymizer.tree_.value[node][0]
                # label = int(self.anonymizer.classes_[np.argmax(label_hist)])
                cell = {'label': 1, 'hist': hist, 'id': int(node)}
                cells_by_id[cell['id']] = cell
        self.nodes = leaves
        self._find_representatives(x, x_anonymizer_train, cells_by_id.values())
        return cells_by_id

    def _find_representatives(self, x, x_anonymizer_train, cells):
        # x is original data, x_anonymizer_train is only QIs + 1-hot encoded
        node_ids = self._find_sample_nodes(x_anonymizer_train)
        for cell in cells:
            cell['representative'] = {}
            # get all rows in cell
            indexes = [index for index, node_id in enumerate(node_ids) if node_id == cell['id']]
            # TODO: should we filter only those with majority label? (using hist)
            if type(x) == np.ndarray:
                rows = x[indexes]
            else:  # pandas
                rows = x.iloc[indexes]
            for feature in self.quasi_identifiers:
                if type(x) == np.ndarray:
                    values = rows[:, feature]
                else:  # pandas
                    values = rows.loc[:, feature]
                if self.categorical_features and feature in self.categorical_features:
                    # find most common value
                    cell['representative'][feature] = Counter(values).most_common(1)[0][0]
                else:
                    # find the mean value (per feature)
                    median = np.median(values)
                    min_value = max(values)
                    min_dist = float("inf")
                    for value in values:
                        dist = distance.euclidean(value, median)
                        if dist < min_dist:
                            min_dist = dist
                            min_value = value
                    cell['representative'][feature] = min_value

    def _find_sample_nodes(self, samples):
        paths = self.anonymizer.decision_path(samples).toarray()
        node_set = set(self.nodes)
        return [(list(set([i for i, v in enumerate(p) if v == 1]) & node_set))[0] for p in paths]

    def _find_sample_cells(self, samples, cells_by_id):
        node_ids = self._find_sample_nodes(samples)
        return [cells_by_id[node_id] for node_id in node_ids]

    def _anonymize_data_numpy(self, x, x_anonymizer_train, cells_by_id):
        cells = self._find_sample_cells(x_anonymizer_train, cells_by_id)
        index = 0
        for row in x:
            cell = cells[index]
            index += 1
            for feature in cell['representative']:
                row[feature] = cell['representative'][feature]
        return x

    def _anonymize_data_pandas(self, x, x_anonymizer_train, cells_by_id):
        cells = self._find_sample_cells(x_anonymizer_train, cells_by_id)
        index = 0
        for i, row in x.iterrows():
            cell = cells[index]
            index += 1
            for feature in cell['representative']:
                x.at[i, feature] = cell['representative'][feature]
        return x

    def _modify_categorical_features(self, x):
        encoder = OneHotEncoder()
        one_hot_encoded = encoder.fit_transform(x)
        return one_hot_encoded
