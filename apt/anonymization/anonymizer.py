import numpy as np
import pandas as pd
from scipy.spatial import distance
from collections import Counter

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from apt.utils.datasets import ArrayDataset, DATA_PANDAS_NUMPY_TYPE

from typing import Union, Optional


class Anonymize:
    """
    Class for performing tailored, model-guided anonymization of training datasets for ML models.

    Based on the implementation described in: https://arxiv.org/abs/2007.13086

    :param k: The privacy parameter that determines the number of records that will be indistinguishable from each
              other (when looking at the quasi identifiers). Should be at least 2.
    :type k: int
    :param quasi_identifiers: The features that need to be minimized in case of pandas data, and indexes of features
                              in case of numpy data.
    :type quasi_identifiers: np.ndarray or list
    :param categorical_features: The list of categorical features (if supplied, these featurtes will be one-hot encoded
                                 before using them to train the decision tree model).
    :type categorical_features: list, optional
    :param is_regression: Whether the model is a regression model or not (if False, assumes a classification model).
                          Default is False.
    :type is_regression: list, optional
    :param train_only_QI: The required method to train data set for anonymization. Default is
                          to train the tree on all features.
    :type train_only_QI: boolean, optional
    """

    def __init__(self, k: int, quasi_identifiers: Union[np.ndarray, list], categorical_features: Optional[list] = None,
                 is_regression: Optional[bool] = False, train_only_QI: Optional[bool] = False):
        if k < 2:
            raise ValueError("k should be a positive integer with a value of 2 or higher")
        if quasi_identifiers is None or len(quasi_identifiers) < 1:
            raise ValueError("The list of quasi-identifiers cannot be empty")

        self.k = k
        self.quasi_identifiers = quasi_identifiers
        self.categorical_features = categorical_features
        self.is_regression = is_regression
        self.train_only_QI = train_only_QI
        self.features_names = None
        self.features = None

    def anonymize(self, dataset: ArrayDataset) -> DATA_PANDAS_NUMPY_TYPE:
        """
        Method for performing model-guided anonymization.

        :param dataset: Data wrapper containing the training data for the model and the predictions of the
                        original model on the training data.
        :type dataset: `ArrayDataset`
        :return: The anonymized training dataset as either numpy array or pandas DataFrame (depending on the type of
                 the original data used to create the ArrayDataset).
        """
        if dataset.get_samples().shape[1] != 0:
            self.features = [i for i in range(dataset.get_samples().shape[1])]
        else:
            raise ValueError('No data provided')

        if dataset.features_names is not None:
            self.features_names = dataset.features_names
        else:  # if no names provided, use numbers instead
            self.features_names = self.features

        if not set(self.quasi_identifiers).issubset(set(self.features_names)):
            raise ValueError('Quasi identifiers should bs a subset of the supplied features or indexes in range of '
                             'the data columns')
        if self.categorical_features and not set(self.categorical_features).issubset(set(self.features_names)):
            raise ValueError('Categorical features should bs a subset of the supplied features or indexes in range of '
                             'the data columns')
        self.quasi_identifiers = [i for i, v in enumerate(self.features_names) if v in self.quasi_identifiers]
        if self.categorical_features:
            self.categorical_features = [i for i, v in enumerate(self.features_names) if v in self.categorical_features]

        transformed = self._anonymize(dataset.get_samples().copy(), dataset.get_labels())
        if dataset.is_pandas:
            return pd.DataFrame(transformed, columns=self.features_names)
        else:
            return transformed

    def _anonymize(self, x, y):
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y should have same number of rows")
        if x.dtype.kind not in 'iufc':
            if not self.categorical_features:
                raise ValueError('when supplying an array with non-numeric data, categorical_features must be defined')
            x_prepared = self._modify_categorical_features(x)
        else:
            x_prepared = x
        x_anonymizer_train = x_prepared
        if self.train_only_QI:
            # build DT just on QI features
            x_anonymizer_train = x_prepared[:, self.quasi_identifiers]
        if self.is_regression:
            self._anonymizer = DecisionTreeRegressor(random_state=10, min_samples_split=2, min_samples_leaf=self.k)
        else:
            self._anonymizer = DecisionTreeClassifier(random_state=10, min_samples_split=2, min_samples_leaf=self.k)

        self._anonymizer.fit(x_anonymizer_train, y)
        cells_by_id = self._calculate_cells(x, x_anonymizer_train)
        return self._anonymize_data(x, x_anonymizer_train, cells_by_id)

    def _calculate_cells(self, x, x_anonymizer_train):
        # x is original data, x_anonymizer_train is only QIs + 1-hot encoded
        cells_by_id = {}
        leaves = []
        for node, feature in enumerate(self._anonymizer.tree_.feature):
            if feature == -2:  # leaf node
                leaves.append(node)
                hist = [int(i) for i in self._anonymizer.tree_.value[node][0]]
                # TODO we may change the method for choosing representative for cell
                # label_hist = self.anonymizer.tree_.value[node][0]
                # label = int(self.anonymizer.classes_[np.argmax(label_hist)])
                cell = {'label': 1, 'hist': hist, 'id': int(node)}
                cells_by_id[cell['id']] = cell
        self._nodes = leaves
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
            rows = x[indexes]
            for feature in self.quasi_identifiers:
                values = rows[:, feature]
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
        paths = self._anonymizer.decision_path(samples).toarray()
        node_set = set(self._nodes)
        return [(list(set([i for i, v in enumerate(p) if v == 1]) & node_set))[0] for p in paths]

    def _find_sample_cells(self, samples, cells_by_id):
        node_ids = self._find_sample_nodes(samples)
        return [cells_by_id[node_id] for node_id in node_ids]

    def _anonymize_data(self, x, x_anonymizer_train, cells_by_id):
        cells = self._find_sample_cells(x_anonymizer_train, cells_by_id)
        index = 0
        for row in x:
            cell = cells[index]
            index += 1
            for feature in cell['representative']:
                row[feature] = cell['representative'][feature]
        return x

    def _modify_categorical_features(self, x):
        # prepare data for DT
        used_features = self.features
        if self.train_only_QI:
            used_features = self.quasi_identifiers
        numeric_features = [f for f in self.features if f in used_features and f not in self.categorical_features]
        categorical_features = [f for f in self.categorical_features if f in used_features]
        numeric_transformer = Pipeline(
            steps=[('imputer', SimpleImputer(strategy='constant', fill_value=0))]
        )
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )
        encoded = preprocessor.fit_transform(x)
        return encoded
