import pytest
import numpy as np
import pandas as pd
import scipy

from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_diabetes
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from torch import nn, optim, sigmoid, where
from torch.nn import functional
from scipy.special import expit

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

from apt.utils.datasets.datasets import PytorchData
from apt.utils.models.pytorch_model import PyTorchClassifier
from apt.minimization import GeneralizeToRepresentative
from apt.utils.dataset_utils import get_iris_dataset_np, get_adult_dataset_pd, get_german_credit_dataset_pd
from apt.utils.datasets import ArrayDataset
from apt.utils.models import SklearnClassifier, SklearnRegressor, KerasClassifier, \
    CLASSIFIER_SINGLE_OUTPUT_CLASS_PROBABILITIES, CLASSIFIER_SINGLE_OUTPUT_CATEGORICAL, \
    CLASSIFIER_SINGLE_OUTPUT_CLASS_LOGITS, CLASSIFIER_MULTI_OUTPUT_BINARY_LOGITS
tf.compat.v1.disable_eager_execution()


ACCURACY_DIFF = 0.05


@pytest.fixture
def diabetes_dataset():
    return load_diabetes()


@pytest.fixture
def cells():
    cells = [{"id": 1, "ranges": {"age": {"start": None, "end": 38}, "height": {"start": None, "end": 170}}, "label": 0,
              'categories': {}, "representative": {"age": 26, "height": 149}},
             {"id": 2, "ranges": {"age": {"start": 39, "end": None}, "height": {"start": None, "end": 170}}, "label": 1,
              'categories': {}, "representative": {"age": 58, "height": 163}},
             {"id": 3, "ranges": {"age": {"start": None, "end": 38}, "height": {"start": 171, "end": None}}, "label": 0,
              'categories': {}, "representative": {"age": 31, "height": 184}},
             {"id": 4, "ranges": {"age": {"start": 39, "end": None}, "height": {"start": 171, "end": None}}, "label": 1,
              'categories': {}, "representative": {"age": 45, "height": 176}}
             ]
    features = ['age', 'height']
    x = np.array([[23, 165],
                  [45, 158],
                  [18, 190]])
    y = [1, 1, 0]
    return cells, features, x, y


@pytest.fixture
def cells_categorical():
    cells = [{'id': 1, 'label': 0, 'ranges': {'age': {'start': None, 'end': None}},
              'categories': {'sex': ['f', 'm']}, 'hist': [2, 0],
              'representative': {'age': 45, 'height': 149, 'sex': 'f'},
              'untouched': ['height']},
             {'id': 3, 'label': 1, 'ranges': {'age': {'start': None, 'end': None}},
              'categories': {'sex': ['f', 'm']}, 'hist': [0, 3],
              'representative': {'age': 23, 'height': 165, 'sex': 'f'},
              'untouched': ['height']},
             {'id': 4, 'label': 0, 'ranges': {'age': {'start': None, 'end': None}},
              'categories': {'sex': ['f', 'm']}, 'hist': [1, 0],
              'representative': {'age': 18, 'height': 190, 'sex': 'm'},
              'untouched': ['height']}
             ]
    features = ['age', 'height', 'sex']
    x = [[23, 165, 'f'],
         [45, 158, 'f'],
         [56, 123, 'f'],
         [67, 154, 'm'],
         [45, 149, 'f'],
         [42, 166, 'm'],
         [73, 172, 'm'],
         [94, 168, 'f'],
         [69, 175, 'm'],
         [24, 181, 'm'],
         [18, 190, 'm']]
    y = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
    return cells, features, x, y


@pytest.fixture
def data_two_features():
    x = np.array([[23, 165],
                  [45, 158],
                  [56, 123],
                  [67, 154],
                  [45, 149],
                  [42, 166],
                  [73, 172],
                  [94, 168],
                  [69, 175],
                  [24, 181],
                  [18, 190]])
    y = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
    x1 = np.array([[33, 165],
                   [43, 150],
                   [71, 143],
                   [92, 194],
                   [13, 125],
                   [22, 169]])
    features = ['age', 'height']
    return x, y, features, x1


@pytest.fixture
def data_three_features():
    features = ['age', 'height', 'weight']
    x = np.array([[23, 165, 70],
                  [45, 158, 67],
                  [56, 123, 65],
                  [67, 154, 90],
                  [45, 149, 67],
                  [42, 166, 58],
                  [73, 172, 68],
                  [94, 168, 69],
                  [69, 175, 80],
                  [24, 181, 95],
                  [18, 190, 102]])
    y = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
    return x, y, features


@pytest.fixture
def data_four_features():
    features = ['age', 'height', 'sex', 'ola']
    x = [[23, 165, 'f', 'aa'],
         [45, 158, 'f', 'aa'],
         [56, 123, 'f', 'bb'],
         [67, 154, 'm', 'aa'],
         [45, 149, 'f', 'bb'],
         [42, 166, 'm', 'bb'],
         [73, 172, 'm', 'bb'],
         [94, 168, 'f', 'aa'],
         [69, 175, 'm', 'aa'],
         [24, 181, 'm', 'bb'],
         [18, 190, 'm', 'bb']]
    y = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
    x1 = [[33, 165, 'f', 'aa'],
          [43, 150, 'm', 'aa'],
          [71, 143, 'f', 'aa'],
          [92, 194, 'm', 'aa'],
          [13, 125, 'f', 'aa'],
          [22, 169, 'f', 'bb']]
    return x, y, features, x1


@pytest.fixture
def data_five_features():
    features = ['age', 'height', 'weight', 'sex', 'ola']
    x = [[23, 165, 65, 'f', 'aa'],
         [45, 158, 76, 'f', 'aa'],
         [56, 123, 78, 'f', 'bb'],
         [67, 154, 87, 'm', 'aa'],
         [45, 149, 45, 'f', 'bb'],
         [42, 166, 76, 'm', 'bb'],
         [73, 172, 85, 'm', 'bb'],
         [94, 168, 92, 'f', 'aa'],
         [69, 175, 95, 'm', 'aa'],
         [24, 181, 49, 'm', 'bb'],
         [18, 190, 69, 'm', 'bb']]
    y = pd.Series([1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
    return x, y, features


def compare_generalizations(gener, expected_generalizations):
    for key in expected_generalizations['ranges']:
        assert (set(expected_generalizations['ranges'][key]) == set(gener['ranges'][key]))
    for key in expected_generalizations['categories']:
        assert (set([frozenset(sl) for sl in expected_generalizations['categories'][key]])
                == set([frozenset(sl) for sl in gener['categories'][key]]))
    assert (set(expected_generalizations['untouched']) == set(gener['untouched']))
    if 'range_representatives' in expected_generalizations:
        for key in expected_generalizations['range_representatives']:
            assert (set(expected_generalizations['range_representatives'][key])
                    == set(gener['range_representatives'][key]))
    if 'category_representatives' in expected_generalizations:
        for key in expected_generalizations['category_representatives']:
            assert (set(expected_generalizations['category_representatives'][key])
                    == set(gener['category_representatives'][key]))


def check_features(features, expected_generalizations, transformed, x, pandas=False):
    modified_features = [f for f in features if
                         f in expected_generalizations['categories'].keys() or f in expected_generalizations[
                             'ranges'].keys()]

    if pandas:
        np.testing.assert_array_equal(transformed.drop(modified_features, axis=1), x.drop(modified_features, axis=1))
        if len(expected_generalizations['ranges'].keys()) > 0 or len(expected_generalizations['categories'].keys()) > 0:
            assert (((transformed[modified_features]).equals(x[modified_features])) is False)
    else:
        indexes = []
        for i in range(len(features)):
            if features[i] in modified_features:
                indexes.append(i)
        if len(indexes) != transformed.shape[1]:
            assert (np.array_equal(np.delete(transformed, indexes, axis=1), np.delete(x, indexes, axis=1)))
        if len(expected_generalizations['ranges'].keys()) > 0 or len(expected_generalizations['categories'].keys()) > 0:
            assert (not np.array_equal(transformed[:, indexes], x[:, indexes]))


def check_ncp(ncp, expected_generalizations):
    if len(expected_generalizations['ranges'].keys()) > 0 or len(expected_generalizations['categories'].keys()) > 0:
        assert (ncp > 0.0)


def test_minimizer_params(cells):
    # Assume two features, age and height, and boolean label
    cells, features, x, y = cells

    base_est = DecisionTreeClassifier(random_state=0, min_samples_split=2,
                                      min_samples_leaf=1)
    model = SklearnClassifier(base_est, CLASSIFIER_SINGLE_OUTPUT_CLASS_PROBABILITIES)
    model.fit(ArrayDataset(x, y))

    expected_generalizations = {'categories': {}, 'category_representatives': {},
                                'range_representatives': {'age': [38, 0.5, 40], 'height': [170, 0.5, 172]},
                                'ranges': {'age': [38, 39], 'height': [170, 171]}, 'untouched': []}

    gen = GeneralizeToRepresentative(model, cells=cells)
    gener = gen.generalizations
    compare_generalizations(gener, expected_generalizations)
    gen.fit()
    gen.transform(dataset=ArrayDataset(x, features_names=features))


def create_encoder(numeric_features, categorical_features, x):
    numeric_transformer = Pipeline(
        steps=[('imputer', SimpleImputer(strategy='constant', fill_value=0))]
    )

    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    encoded = preprocessor.fit_transform(x)
    if scipy.sparse.issparse(encoded):
        pd.DataFrame.sparse.from_spmatrix(encoded)
    else:
        encoded = pd.DataFrame(encoded)

    return preprocessor, encoded


def test_minimizer_params_not_transform(cells):
    # Assume two features, age and height, and boolean label
    cells, features, x, y = cells
    samples = ArrayDataset(x, y, features)
    base_est = DecisionTreeClassifier(random_state=0, min_samples_split=2,
                                      min_samples_leaf=1)
    model = SklearnClassifier(base_est, CLASSIFIER_SINGLE_OUTPUT_CLASS_PROBABILITIES)
    model.fit(ArrayDataset(x, y))

    gen = GeneralizeToRepresentative(model, cells=cells, generalize_using_transform=False)
    ncp = gen.calculate_ncp(samples)
    assert (ncp > 0.0)


def test_minimizer_fit(data_two_features):
    x, y, features, _ = data_two_features
    base_est = DecisionTreeClassifier(random_state=0, min_samples_split=2,
                                      min_samples_leaf=1)
    model = SklearnClassifier(base_est, CLASSIFIER_SINGLE_OUTPUT_CLASS_PROBABILITIES)
    model.fit(ArrayDataset(x, y))
    ad = ArrayDataset(x)
    predictions = model.predict(ad)
    if predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)
    target_accuracy = 0.5
    gen = GeneralizeToRepresentative(model, target_accuracy=target_accuracy)
    train_dataset = ArrayDataset(x, predictions, features_names=features)

    gen.fit(dataset=train_dataset)
    transformed = gen.transform(dataset=ad)
    gener = gen.generalizations
    expected_generalizations = {'ranges': {}, 'categories': {}, 'untouched': ['height', 'age']}

    compare_generalizations(gener, expected_generalizations)
    check_features(features, expected_generalizations, transformed, x)
    assert (np.equal(x, transformed).all())
    ncp = gen.ncp.transform_score
    check_ncp(ncp, expected_generalizations)

    rel_accuracy = model.score(ArrayDataset(transformed, predictions))
    assert ((rel_accuracy >= target_accuracy) or (target_accuracy - rel_accuracy) <= ACCURACY_DIFF)


def test_minimizer_ncp(data_two_features):
    x, y, features, x1 = data_two_features

    base_est = DecisionTreeClassifier(random_state=0, min_samples_split=2,
                                      min_samples_leaf=1)
    model = SklearnClassifier(base_est, CLASSIFIER_SINGLE_OUTPUT_CLASS_PROBABILITIES)
    model.fit(ArrayDataset(x, y))
    ad = ArrayDataset(x)
    ad1 = ArrayDataset(x1, features_names=features)
    predictions = model.predict(ad)
    if predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)
    target_accuracy = 0.4
    train_dataset = ArrayDataset(x, predictions, features_names=features)

    gen1 = GeneralizeToRepresentative(model, target_accuracy=target_accuracy, generalize_using_transform=False)
    gen1.fit(dataset=train_dataset)
    ncp1 = gen1.ncp.fit_score
    ncp2 = gen1.calculate_ncp(ad1)

    gen2 = GeneralizeToRepresentative(model, target_accuracy=target_accuracy)
    gen2.fit(dataset=train_dataset)
    ncp3 = gen2.ncp.fit_score
    gen2.transform(dataset=ad1)
    ncp4 = gen2.ncp.transform_score
    gen2.transform(dataset=ad)
    ncp5 = gen2.ncp.transform_score
    gen2.transform(dataset=ad1)
    ncp6 = gen2.ncp.transform_score

    assert (ncp1 <= ncp3)
    assert (ncp2 != ncp3)
    assert (ncp3 != ncp4)
    assert (ncp4 != ncp5)
    assert (ncp6 == ncp4)


def test_minimizer_ncp_categorical(data_four_features):
    x, y, features, x1 = data_four_features
    x = pd.DataFrame(x, columns=features)
    x1 = pd.DataFrame(x1, columns=features)

    numeric_features = ["age", "height"]
    categorical_features = ["sex", "ola"]
    preprocessor, encoded = create_encoder(numeric_features, categorical_features, x)

    base_est = DecisionTreeClassifier(random_state=0, min_samples_split=2,
                                      min_samples_leaf=1)
    model = SklearnClassifier(base_est, CLASSIFIER_SINGLE_OUTPUT_CLASS_PROBABILITIES)
    model.fit(ArrayDataset(encoded, y))
    ad = ArrayDataset(x)
    ad1 = ArrayDataset(x1)
    predictions = model.predict(ArrayDataset(encoded))
    if predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)
    target_accuracy = 0.4
    train_dataset = ArrayDataset(x, predictions, features_names=features)

    gen1 = GeneralizeToRepresentative(model, target_accuracy=target_accuracy,
                                      categorical_features=categorical_features,
                                      generalize_using_transform=False,
                                      encoder=preprocessor)
    gen1.fit(dataset=train_dataset)
    ncp1 = gen1.ncp.fit_score
    ncp2 = gen1.calculate_ncp(ad1)

    gen2 = GeneralizeToRepresentative(model, target_accuracy=target_accuracy, categorical_features=categorical_features,
                                      encoder=preprocessor)
    gen2.fit(dataset=train_dataset)
    ncp3 = gen2.ncp.fit_score
    gen2.transform(dataset=ad1)
    ncp4 = gen2.ncp.transform_score
    gen2.transform(dataset=ad)
    ncp5 = gen2.ncp.transform_score
    gen2.transform(dataset=ad1)
    ncp6 = gen2.ncp.transform_score

    assert (ncp1 <= ncp3)
    assert (ncp2 != ncp3)
    assert (ncp3 != ncp4)
    assert (ncp4 != ncp5)
    assert (ncp6 == ncp4)


def test_minimizer_fit_not_transform(data_two_features):
    x, y, features, x1 = data_two_features
    base_est = DecisionTreeClassifier(random_state=0, min_samples_split=2,
                                      min_samples_leaf=1)
    model = SklearnClassifier(base_est, CLASSIFIER_SINGLE_OUTPUT_CLASS_PROBABILITIES)
    model.fit(ArrayDataset(x, y))
    ad = ArrayDataset(x)
    predictions = model.predict(ad)
    if predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)
    target_accuracy = 0.5
    gen = GeneralizeToRepresentative(model, target_accuracy=target_accuracy, generalize_using_transform=False)
    train_dataset = ArrayDataset(x, predictions, features_names=features)

    gen.fit(dataset=train_dataset)
    gener = gen.generalizations
    expected_generalizations = {'ranges': {'age': [], 'height': [157.0]}, 'categories': {}, 'untouched': []}

    compare_generalizations(gener, expected_generalizations)

    ncp = gen.ncp.fit_score
    check_ncp(ncp, expected_generalizations)


def test_minimizer_fit_pandas(data_four_features):
    x, y, features, _ = data_four_features
    x = pd.DataFrame(x, columns=features)

    numeric_features = ["age", "height"]
    categorical_features = ["sex", "ola"]
    preprocessor, encoded = create_encoder(numeric_features, categorical_features, x)

    base_est = DecisionTreeClassifier(random_state=0, min_samples_split=2,
                                      min_samples_leaf=1)
    model = SklearnClassifier(base_est, CLASSIFIER_SINGLE_OUTPUT_CLASS_PROBABILITIES)
    model.fit(ArrayDataset(encoded, y))
    predictions = model.predict(ArrayDataset(encoded))
    if predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)

    # Append classifier to preprocessing pipeline.
    # Now we have a full prediction pipeline.
    target_accuracy = 0.5
    gen = GeneralizeToRepresentative(model, target_accuracy=target_accuracy,
                                     categorical_features=categorical_features,
                                     encoder=preprocessor)
    train_dataset = ArrayDataset(x, predictions)
    gen.fit(dataset=train_dataset)
    transformed = gen.transform(dataset=ArrayDataset(x))
    gener = gen.generalizations
    expected_generalizations = {'ranges': {'age': []}, 'categories': {},
                                'untouched': ['height', 'sex', 'ola']}

    compare_generalizations(gener, expected_generalizations)
    check_features(features, expected_generalizations, transformed, x, True)
    ncp = gen.ncp.transform_score
    check_ncp(ncp, expected_generalizations)

    rel_accuracy = model.score(ArrayDataset(preprocessor.transform(transformed), predictions))
    assert ((rel_accuracy >= target_accuracy) or (target_accuracy - rel_accuracy) <= ACCURACY_DIFF)


def test_minimizer_params_categorical(cells_categorical):
    # Assume three features, age, sex and height, and boolean label
    cells, features, x, y = cells_categorical

    x = pd.DataFrame(x, columns=features)
    numeric_features = ["age", "height"]
    categorical_features = ["sex"]
    preprocessor, encoded = create_encoder(numeric_features, categorical_features, x)
    base_est = DecisionTreeClassifier(random_state=0, min_samples_split=2,
                                      min_samples_leaf=1)
    model = SklearnClassifier(base_est, CLASSIFIER_SINGLE_OUTPUT_CLASS_PROBABILITIES)
    model.fit(ArrayDataset(encoded, y))
    predictions = model.predict(ArrayDataset(encoded))
    if predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)
    # Append classifier to preprocessing pipeline.
    # Now we have a full prediction pipeline.
    target_accuracy = 0.5
    gen = GeneralizeToRepresentative(model, target_accuracy=target_accuracy,
                                     categorical_features=categorical_features, cells=cells,
                                     encoder=preprocessor)
    train_dataset = ArrayDataset(x, predictions)
    gen.fit(dataset=train_dataset)
    transformed = gen.transform(dataset=ArrayDataset(x))

    rel_accuracy = model.score(ArrayDataset(preprocessor.transform(transformed), predictions))
    assert ((rel_accuracy >= target_accuracy) or (target_accuracy - rel_accuracy) <= ACCURACY_DIFF)


def test_minimizer_fit_qi(data_three_features):
    x, y, features = data_three_features
    qi = ['age', 'weight']
    base_est = DecisionTreeClassifier(random_state=0, min_samples_split=2,
                                      min_samples_leaf=1)
    model = SklearnClassifier(base_est, CLASSIFIER_SINGLE_OUTPUT_CLASS_PROBABILITIES)
    model.fit(ArrayDataset(x, y))
    ad = ArrayDataset(x)
    predictions = model.predict(ad)
    if predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)
    target_accuracy = 0.5
    gen = GeneralizeToRepresentative(model, target_accuracy=target_accuracy, features_to_minimize=qi)
    train_dataset = ArrayDataset(x, predictions, features_names=features)
    gen.fit(dataset=train_dataset)
    transformed = gen.transform(dataset=ad)
    gener = gen.generalizations
    expected_generalizations = {'ranges': {'age': [], 'weight': [67.5]}, 'categories': {}, 'untouched': ['height']}
    compare_generalizations(gener, expected_generalizations)
    check_features(features, expected_generalizations, transformed, x)
    assert ((np.delete(transformed, [0, 2], axis=1) == np.delete(x, [0, 2], axis=1)).all())
    ncp = gen.ncp.transform_score
    check_ncp(ncp, expected_generalizations)

    rel_accuracy = model.score(ArrayDataset(transformed, predictions))
    assert ((rel_accuracy >= target_accuracy) or (target_accuracy - rel_accuracy) <= ACCURACY_DIFF)


def test_minimizer_fit_pandas_qi(data_five_features):
    x, y, features = data_five_features
    x = pd.DataFrame(x, columns=features)
    qi = ['age', 'weight', 'ola']

    numeric_features = ["age", "height", "weight"]
    categorical_features = ["sex", "ola"]
    preprocessor, encoded = create_encoder(numeric_features, categorical_features, x)

    base_est = DecisionTreeClassifier(random_state=0, min_samples_split=2,
                                      min_samples_leaf=1)
    model = SklearnClassifier(base_est, CLASSIFIER_SINGLE_OUTPUT_CLASS_PROBABILITIES)
    model.fit(ArrayDataset(encoded, y))
    predictions = model.predict(ArrayDataset(encoded))
    if predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)

    # Append classifier to preprocessing pipeline.
    # Now we have a full prediction pipeline.
    target_accuracy = 0.5
    gen = GeneralizeToRepresentative(model, target_accuracy=target_accuracy,
                                     categorical_features=categorical_features, features_to_minimize=qi,
                                     encoder=preprocessor)
    train_dataset = ArrayDataset(x, predictions)
    gen.fit(dataset=train_dataset)
    transformed = gen.transform(dataset=ArrayDataset(x))
    gener = gen.generalizations
    expected_generalizations = {'ranges': {'age': [], 'weight': [47.0]}, 'categories': {'ola': [['bb', 'aa']]},
                                'untouched': ['height', 'sex']}

    compare_generalizations(gener, expected_generalizations)
    check_features(features, expected_generalizations, transformed, x, True)
    np.testing.assert_array_equal(transformed.drop(qi, axis=1), x.drop(qi, axis=1))
    ncp = gen.ncp.transform_score
    check_ncp(ncp, expected_generalizations)

    rel_accuracy = model.score(ArrayDataset(preprocessor.transform(transformed), predictions))
    assert ((rel_accuracy >= target_accuracy) or (target_accuracy - rel_accuracy) <= ACCURACY_DIFF)


def test_minimize_ndarray_iris():
    features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    (x_train, y_train), _ = get_iris_dataset_np()
    qi = ['sepal length (cm)', 'petal length (cm)']
    base_est = DecisionTreeClassifier(random_state=0, min_samples_split=2,
                                      min_samples_leaf=1)
    model = SklearnClassifier(base_est, CLASSIFIER_SINGLE_OUTPUT_CATEGORICAL)
    model.fit(ArrayDataset(x_train, y_train))
    predictions = model.predict(ArrayDataset(x_train))
    if predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)
    target_accuracy = 0.3
    gen = GeneralizeToRepresentative(model, target_accuracy=target_accuracy, features_to_minimize=qi)
    transformed = gen.fit_transform(dataset=ArrayDataset(x_train, predictions, features_names=features))
    gener = gen.generalizations
    expected_generalizations = {'ranges': {'sepal length (cm)': [], 'petal length (cm)': [2.449999988079071]},
                                'categories': {}, 'untouched': ['petal width (cm)', 'sepal width (cm)']}

    compare_generalizations(gener, expected_generalizations)
    assert ((np.delete(transformed, [0, 2], axis=1) == np.delete(x_train, [0, 2], axis=1)).all())

    check_features(features, expected_generalizations, transformed, x_train)
    ncp = gen.ncp.transform_score
    check_ncp(ncp, expected_generalizations)

    rel_accuracy = model.score(ArrayDataset(transformed, predictions))
    assert ((rel_accuracy >= target_accuracy) or (target_accuracy - rel_accuracy) <= ACCURACY_DIFF)


def test_minimize_pandas_adult():
    (x_train, y_train), _ = get_adult_dataset_pd()
    x_train = x_train.head(1000)
    y_train = y_train.head(1000)

    features = ['age', 'workclass', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
    x_train = pd.DataFrame(x_train, columns=features)

    categorical_features = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                            'hours-per-week', 'native-country']

    qi = ['age', 'workclass', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
          'native-country']

    numeric_features = [f for f in features if f not in categorical_features]
    preprocessor, encoded = create_encoder(numeric_features, categorical_features, x_train)

    base_est = DecisionTreeClassifier(random_state=0, min_samples_split=2,
                                      min_samples_leaf=1)
    model = SklearnClassifier(base_est, CLASSIFIER_SINGLE_OUTPUT_CLASS_PROBABILITIES)
    model.fit(ArrayDataset(encoded, y_train))
    predictions = model.predict(ArrayDataset(encoded))
    if predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)
    target_accuracy = 0.7
    gen = GeneralizeToRepresentative(model, target_accuracy=target_accuracy,
                                     categorical_features=categorical_features, features_to_minimize=qi,
                                     encoder=preprocessor)
    gen.fit(dataset=ArrayDataset(x_train, predictions, features_names=features))
    transformed = gen.transform(dataset=ArrayDataset(x_train))
    gener = gen.generalizations
    expected_generalizations = {'ranges': {'age': [], 'education-num': []}, 'categories': {
        'workclass': [['Self-emp-not-inc', 'Private', 'Federal-gov', 'Self-emp-inc', '?', 'Local-gov', 'State-gov']],
        'marital-status': [
            ['Divorced', 'Married-AF-spouse', 'Married-spouse-absent', 'Widowed', 'Separated', 'Married-civ-spouse',
             'Never-married']], 'occupation': [
            ['Tech-support', 'Priv-house-serv', 'Machine-op-inspct', 'Other-service', 'Prof-specialty', 'Adm-clerical',
             'Protective-serv', 'Handlers-cleaners', 'Transport-moving', 'Armed-Forces', '?', 'Sales',
             'Farming-fishing', 'Exec-managerial', 'Craft-repair']],
        'relationship': [['Not-in-family', 'Wife', 'Other-relative', 'Husband', 'Unmarried', 'Own-child']],
        'race': [['Asian-Pac-Islander', 'White', 'Other', 'Black', 'Amer-Indian-Eskimo']], 'sex': [['Female', 'Male']],
        'native-country': [
            ['Euro_1', 'LatinAmerica', 'BritishCommonwealth', 'SouthAmerica', 'UnitedStates', 'China', 'Euro_2',
             'SE_Asia', 'Other', 'Unknown']]}, 'untouched': ['capital-loss', 'hours-per-week', 'capital-gain']}

    compare_generalizations(gener, expected_generalizations)

    np.testing.assert_array_equal(transformed.drop(qi, axis=1), x_train.drop(qi, axis=1))

    check_features(features, expected_generalizations, transformed, x_train, True)
    ncp = gen.ncp.transform_score
    check_ncp(ncp, expected_generalizations)

    rel_accuracy = model.score(ArrayDataset(preprocessor.transform(transformed), predictions))
    assert ((rel_accuracy >= target_accuracy) or (target_accuracy - rel_accuracy) <= ACCURACY_DIFF)


def test_german_credit_pandas():
    (x_train, y_train), _ = get_german_credit_dataset_pd()
    features = ["Existing_checking_account", "Duration_in_month", "Credit_history", "Purpose", "Credit_amount",
                "Savings_account", "Present_employment_since", "Installment_rate", "Personal_status_sex", "debtors",
                "Present_residence", "Property", "Age", "Other_installment_plans", "Housing",
                "Number_of_existing_credits", "Job", "N_people_being_liable_provide_maintenance", "Telephone",
                "Foreign_worker"]
    categorical_features = ["Existing_checking_account", "Credit_history", "Purpose", "Savings_account",
                            "Present_employment_since", "Personal_status_sex", "debtors", "Property",
                            "Other_installment_plans", "Housing", "Job"]
    qi = ["Duration_in_month", "Credit_history", "Purpose", "debtors", "Property", "Other_installment_plans",
          "Housing", "Job"]

    numeric_features = [f for f in features if f not in categorical_features]
    preprocessor, encoded = create_encoder(numeric_features, categorical_features, x_train)

    base_est = DecisionTreeClassifier(random_state=0, min_samples_split=2,
                                      min_samples_leaf=1)
    model = SklearnClassifier(base_est, CLASSIFIER_SINGLE_OUTPUT_CLASS_PROBABILITIES)
    model.fit(ArrayDataset(encoded, y_train))
    predictions = model.predict(ArrayDataset(encoded))
    if predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)
    target_accuracy = 0.7
    gen = GeneralizeToRepresentative(model, target_accuracy=target_accuracy,
                                     categorical_features=categorical_features, features_to_minimize=qi,
                                     encoder=preprocessor)
    gen.fit(dataset=ArrayDataset(x_train, predictions))
    transformed = gen.transform(dataset=ArrayDataset(x_train))
    gener = gen.generalizations
    expected_generalizations = {'ranges': {'Duration_in_month': [31.5]},
                                'categories': {'Credit_history': [['A30', 'A32', 'A31', 'A34', 'A33']],
                                               'Purpose': [
                                                   ['A41', 'A46', 'A43', 'A40', 'A44', 'A410', 'A49', 'A45', 'A48',
                                                    'A42']],
                                               'debtors': [['A101', 'A102', 'A103']],
                                               'Property': [['A124', 'A121', 'A122', 'A123']],
                                               'Other_installment_plans': [['A142', 'A141', 'A143']],
                                               'Housing': [['A151', 'A152', 'A153']],
                                               'Job': [['A172', 'A171', 'A174', 'A173']]},
                                'untouched': ['Installment_rate', 'Present_residence', 'Personal_status_sex',
                                              'Foreign_worker', 'Telephone', 'Savings_account',
                                              'Number_of_existing_credits', 'N_people_being_liable_provide_maintenance',
                                              'Age', 'Existing_checking_account', 'Credit_amount',
                                              'Present_employment_since']}

    compare_generalizations(gener, expected_generalizations)

    np.testing.assert_array_equal(transformed.drop(qi, axis=1), x_train.drop(qi, axis=1))

    check_features(features, expected_generalizations, transformed, x_train, True)
    ncp = gen.ncp.transform_score
    check_ncp(ncp, expected_generalizations)

    rel_accuracy = model.score(ArrayDataset(preprocessor.transform(transformed), predictions))
    assert ((rel_accuracy >= target_accuracy) or (target_accuracy - rel_accuracy) <= ACCURACY_DIFF)


def test_regression(diabetes_dataset):
    x_train, x_test, y_train, y_test = train_test_split(diabetes_dataset.data, diabetes_dataset.target, test_size=0.5,
                                                        random_state=14)

    base_est = DecisionTreeRegressor(random_state=10, min_samples_split=2)
    model = SklearnRegressor(base_est)
    model.fit(ArrayDataset(x_train, y_train))
    predictions = model.predict(ArrayDataset(x_train))
    qi = ['age', 'bmi', 's2', 's5']
    features = ['age', 'sex', 'bmi', 'bp',
                's1', 's2', 's3', 's4', 's5', 's6']

    target_accuracy = 0.7
    gen = GeneralizeToRepresentative(model, target_accuracy=target_accuracy, is_regression=True,
                                     features_to_minimize=qi)
    gen.fit(dataset=ArrayDataset(x_train, predictions, features_names=features))
    transformed = gen.transform(dataset=ArrayDataset(x_train, features_names=features))
    print('Base model accuracy (R2 score): ', model.score(ArrayDataset(x_test, y_test)))
    model.fit(ArrayDataset(transformed, y_train))
    print('Base model accuracy (R2 score) after anonymization: ', model.score(ArrayDataset(x_test, y_test)))
    gener = gen.generalizations
    expected_generalizations = {'ranges': {
        'age': [-0.07816532626748085, -0.07090024650096893, -0.05637009255588055, -0.05092128552496433,
                -0.04728874587453902, -0.04547247663140297, -0.04183994047343731, -0.027309784665703773,
                -0.023677248042076826, -0.020044708624482155, -0.01641217083670199, -0.001882016600575298,
                0.0017505218856967986, 0.0035667913616634905, 0.007199329789727926, 0.010831868276000023,
                0.02354575227946043, 0.030810829252004623, 0.03262709779664874, 0.03444336913526058,
                0.03625963814556599, 0.03807590529322624, 0.03807590715587139, 0.047157252207398415,
                0.06168740428984165, 0.0635036751627922, 0.06895248219370842, 0.07258502021431923, 0.07621755823493004,
                0.1034616008400917],
        'bmi': [-0.07626373693346977, -0.060635464265942574, -0.056863121688365936, -0.05578530766069889,
                -0.054168591275811195, -0.042312657460570335, -0.0374625027179718, -0.03422906715422869,
                -0.033690162003040314, -0.03261234890669584, -0.02614547684788704, -0.025067666545510292,
                -0.022373135201632977, -0.016984074376523495, -0.01375063881278038, -0.007822672137990594,
                -0.004589236050378531, 0.008344509289599955, 0.015889193629845977, 0.016967005096375942,
                0.024511689320206642, 0.0272062208969146, 0.030978563241660595, 0.032595280557870865,
                0.033673093654215336, 0.04391230642795563, 0.04552902653813362, 0.05469042807817459,
                0.06977979838848114, 0.07301323488354683, 0.09349166229367256],
        's2': [-0.1044962927699089, -0.08649025857448578, -0.07740895450115204, -0.07114598527550697,
               -0.06378699466586113, -0.05971606448292732, -0.04437179118394852, -0.0398311372846365,
               -0.03137612994760275, -0.022138250060379505, -0.018067320343106985, -0.017910746857523918,
               -0.017910745926201344, -0.01618842873722315, -0.007576846517622471, -0.007263698382303119,
               -0.0010007291566580534, 0.0010347360512241721, 0.006514834007248282, 0.00933317095041275,
               0.012464655097573996, 0.019197346206055954, 0.020919663831591606, 0.02217225730419159,
               0.032036433927714825, 0.036420512944459915, 0.04080459102988243, 0.04127431474626064,
               0.04268348217010498, 0.04424922354519367, 0.04424922540783882, 0.056462014093995094, 0.05928034894168377,
               0.061315815430134535, 0.06272498145699501, 0.06460387445986271]}, 'categories': {},
        'untouched': ['s5', 's3', 'bp', 's1', 'sex', 's6', 's4']}

    compare_generalizations(gener, expected_generalizations)
    assert ((np.delete(transformed, [0, 2, 5, 8], axis=1) == np.delete(x_train, [0, 2, 5, 8], axis=1)).all())

    check_features(features, expected_generalizations, transformed, x_train)
    ncp = gen.ncp.transform_score
    check_ncp(ncp, expected_generalizations)

    rel_accuracy = model.score(ArrayDataset(transformed, predictions))
    assert ((rel_accuracy >= target_accuracy) or (target_accuracy - rel_accuracy) <= ACCURACY_DIFF)


def test_x_y():
    features = ['0', '1', '2']
    x = np.array([[23, 165, 70],
                  [45, 158, 67],
                  [56, 123, 65],
                  [67, 154, 90],
                  [45, 149, 67],
                  [42, 166, 58],
                  [73, 172, 68],
                  [94, 168, 69],
                  [69, 175, 80],
                  [24, 181, 95],
                  [18, 190, 102]])
    print(x)
    y = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
    qi = [0, 2]
    base_est = DecisionTreeClassifier(random_state=0, min_samples_split=2,
                                      min_samples_leaf=1)
    model = SklearnClassifier(base_est, CLASSIFIER_SINGLE_OUTPUT_CLASS_PROBABILITIES)
    model.fit(ArrayDataset(x, y))
    ad = ArrayDataset(x)
    predictions = model.predict(ad)
    if predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)
    target_accuracy = 0.5
    gen = GeneralizeToRepresentative(model, target_accuracy=target_accuracy, features_to_minimize=qi)
    gen.fit(X=x, y=predictions)
    transformed = gen.transform(x)
    gener = gen.generalizations
    expected_generalizations = {'ranges': {'0': [], '2': [67.5]}, 'categories': {}, 'untouched': ['1']}
    compare_generalizations(gener, expected_generalizations)
    assert ((np.delete(transformed, [0, 2], axis=1) == np.delete(x, [0, 2], axis=1)).all())
    check_features(features, expected_generalizations, transformed, x)
    ncp = gen.ncp.transform_score
    check_ncp(ncp, expected_generalizations)

    rel_accuracy = model.score(ArrayDataset(transformed, predictions))
    assert ((rel_accuracy >= target_accuracy) or (target_accuracy - rel_accuracy) <= ACCURACY_DIFF)


def test_x_y_features_names():
    features = ['age', 'height', 'weight']
    x = np.array([[23, 165, 70],
                  [45, 158, 67],
                  [56, 123, 65],
                  [67, 154, 90],
                  [45, 149, 67],
                  [42, 166, 58],
                  [73, 172, 68],
                  [94, 168, 69],
                  [69, 175, 80],
                  [24, 181, 95],
                  [18, 190, 102]])
    print(x)
    y = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
    qi = ['age', 'weight']
    base_est = DecisionTreeClassifier(random_state=0, min_samples_split=2,
                                      min_samples_leaf=1)
    model = SklearnClassifier(base_est, CLASSIFIER_SINGLE_OUTPUT_CLASS_PROBABILITIES)
    model.fit(ArrayDataset(x, y))
    ad = ArrayDataset(x)
    predictions = model.predict(ad)
    if predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)
    target_accuracy = 0.5
    gen = GeneralizeToRepresentative(model, target_accuracy=target_accuracy, features_to_minimize=qi)
    gen.fit(X=x, y=predictions, features_names=features)
    transformed = gen.transform(X=x, features_names=features)
    gener = gen.generalizations
    expected_generalizations = {'ranges': {'age': [], 'weight': [67.5]}, 'categories': {}, 'untouched': ['height']}
    compare_generalizations(gener, expected_generalizations)
    assert ((np.delete(transformed, [0, 2], axis=1) == np.delete(x, [0, 2], axis=1)).all())
    check_features(features, expected_generalizations, transformed, x)
    ncp = gen.ncp.transform_score
    check_ncp(ncp, expected_generalizations)

    rel_accuracy = model.score(ArrayDataset(transformed, predictions))
    assert ((rel_accuracy >= target_accuracy) or (target_accuracy - rel_accuracy) <= ACCURACY_DIFF)


def test_BaseEstimator_classification(data_five_features):
    x, y, features = data_five_features
    x = pd.DataFrame(x, columns=features)
    QI = ['age', 'weight', 'ola']

    numeric_features = ["age", "height", "weight"]
    categorical_features = ["sex", "ola"]
    preprocessor, encoded = create_encoder(numeric_features, categorical_features, x)

    base_est = DecisionTreeClassifier(random_state=0, min_samples_split=2,
                                      min_samples_leaf=1)
    model = base_est
    model.fit(encoded, y)
    predictions = model.predict(encoded)

    # Append classifier to preprocessing pipeline.
    # Now we have a full prediction pipeline.
    target_accuracy = 0.5
    gen = GeneralizeToRepresentative(model, target_accuracy=target_accuracy,
                                     categorical_features=categorical_features, features_to_minimize=QI,
                                     encoder=preprocessor)
    train_dataset = ArrayDataset(x, predictions)
    gen.fit(dataset=train_dataset)
    transformed = gen.transform(dataset=ArrayDataset(x))
    gener = gen.generalizations
    expected_generalizations = {'ranges': {'age': [], 'weight': [47.0]}, 'categories': {'ola': [['bb', 'aa']]},
                                'untouched': ['height', 'sex']}

    compare_generalizations(gener, expected_generalizations)

    np.testing.assert_array_equal(transformed.drop(QI, axis=1), x.drop(QI, axis=1))
    check_features(features, expected_generalizations, transformed, x, True)
    ncp = gen.ncp.transform_score
    check_ncp(ncp, expected_generalizations)

    rel_accuracy = model.score(preprocessor.transform(transformed), predictions)
    assert ((rel_accuracy >= target_accuracy) or (target_accuracy - rel_accuracy) <= ACCURACY_DIFF)


def test_BaseEstimator_regression(diabetes_dataset):
    x_train, x_test, y_train, y_test = train_test_split(diabetes_dataset.data, diabetes_dataset.target, test_size=0.5,
                                                        random_state=14)

    base_est = DecisionTreeRegressor(random_state=10, min_samples_split=2)
    model = base_est
    model.fit(x_train, y_train)
    predictions = model.predict(x_train)
    QI = ['age', 'bmi', 's2', 's5']
    features = ['age', 'sex', 'bmi', 'bp',
                's1', 's2', 's3', 's4', 's5', 's6']
    target_accuracy = 0.7
    gen = GeneralizeToRepresentative(model, target_accuracy=target_accuracy, is_regression=True,
                                     features_to_minimize=QI)
    gen.fit(dataset=ArrayDataset(x_train, predictions, features_names=features))
    transformed = gen.transform(dataset=ArrayDataset(x_train, features_names=features))
    print('Base model accuracy (R2 score): ', model.score(x_test, y_test))
    model.fit(transformed, y_train)
    print('Base model accuracy (R2 score) after minimization: ', model.score(x_test, y_test))
    gener = gen.generalizations
    expected_generalizations = {'ranges': {
        'age': [-0.07816532626748085, -0.07090024650096893, -0.05637009255588055, -0.05092128552496433,
                -0.04728874587453902, -0.04547247663140297, -0.04183994047343731, -0.027309784665703773,
                -0.023677248042076826, -0.020044708624482155, -0.01641217083670199, -0.001882016600575298,
                0.0017505218856967986, 0.0035667913616634905, 0.007199329789727926, 0.010831868276000023,
                0.02354575227946043, 0.030810829252004623, 0.03262709779664874, 0.03444336913526058,
                0.03625963814556599, 0.03807590529322624, 0.03807590715587139, 0.047157252207398415,
                0.06168740428984165, 0.0635036751627922, 0.06895248219370842, 0.07258502021431923, 0.07621755823493004,
                0.1034616008400917],
        'bmi': [-0.07626373693346977, -0.060635464265942574, -0.056863121688365936, -0.05578530766069889,
                -0.054168591275811195, -0.042312657460570335, -0.0374625027179718, -0.03422906715422869,
                -0.033690162003040314, -0.03261234890669584, -0.02614547684788704, -0.025067666545510292,
                -0.022373135201632977, -0.016984074376523495, -0.01375063881278038, -0.007822672137990594,
                -0.004589236050378531, 0.008344509289599955, 0.015889193629845977, 0.016967005096375942,
                0.024511689320206642, 0.0272062208969146, 0.030978563241660595, 0.032595280557870865,
                0.033673093654215336, 0.04391230642795563, 0.04552902653813362, 0.05469042807817459,
                0.06977979838848114, 0.07301323488354683, 0.09349166229367256],
        's2': [-0.1044962927699089, -0.08649025857448578, -0.07740895450115204, -0.07114598527550697,
               -0.06378699466586113, -0.05971606448292732, -0.04437179118394852, -0.0398311372846365,
               -0.03137612994760275, -0.022138250060379505, -0.018067320343106985, -0.017910746857523918,
               -0.017910745926201344, -0.01618842873722315, -0.007576846517622471, -0.007263698382303119,
               -0.0010007291566580534, 0.0010347360512241721, 0.006514834007248282, 0.00933317095041275,
               0.012464655097573996, 0.019197346206055954, 0.020919663831591606, 0.02217225730419159,
               0.032036433927714825, 0.036420512944459915, 0.04080459102988243, 0.04127431474626064,
               0.04268348217010498, 0.04424922354519367, 0.04424922540783882, 0.056462014093995094, 0.05928034894168377,
               0.061315815430134535, 0.06272498145699501, 0.06460387445986271]}, 'categories': {},
        'untouched': ['s5', 's3', 'bp', 's1', 'sex', 's6', 's4']}

    compare_generalizations(gener, expected_generalizations)
    assert ((np.delete(transformed, [0, 2, 5, 8], axis=1) == np.delete(x_train, [0, 2, 5, 8], axis=1)).all())

    check_features(features, expected_generalizations, transformed, x_train)
    ncp = gen.ncp.transform_score
    check_ncp(ncp, expected_generalizations)

    rel_accuracy = model.score(transformed, predictions)
    assert ((rel_accuracy >= target_accuracy) or (target_accuracy - rel_accuracy) <= ACCURACY_DIFF)


def test_minimizer_ndarray_one_hot():
    x_train = np.array([[23, 0, 1, 165],
                        [45, 0, 1, 158],
                        [56, 1, 0, 123],
                        [67, 0, 1, 154],
                        [45, 1, 0, 149],
                        [42, 1, 0, 166],
                        [73, 0, 1, 172],
                        [94, 0, 1, 168],
                        [69, 0, 1, 175],
                        [24, 1, 0, 181],
                        [18, 1, 0, 190]])
    y_train = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0])

    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    predictions = model.predict(x_train)

    features = ['0', '1', '2', '3']
    QI = [0, 1, 2]
    QI_slices = [[1, 2]]
    target_accuracy = 0.7
    gen = GeneralizeToRepresentative(model, target_accuracy=target_accuracy, feature_slices=QI_slices,
                                     features_to_minimize=QI)
    gen.fit(dataset=ArrayDataset(x_train, predictions))
    transformed = gen.transform(dataset=ArrayDataset(x_train))
    gener = gen.generalizations
    expected_generalizations = {'categories': {}, 'category_representatives': {},
                                'range_representatives': {'0': [34.5]},
                                'ranges': {'0': [34.5]}, 'untouched': ['3', '1', '2']}

    compare_generalizations(gener, expected_generalizations)

    check_features(features, expected_generalizations, transformed, x_train)
    ncp = gen.ncp.transform_score
    check_ncp(ncp, expected_generalizations)

    rel_accuracy = model.score(transformed, predictions)
    assert ((rel_accuracy >= target_accuracy) or (target_accuracy - rel_accuracy) <= ACCURACY_DIFF)
    transformed_slice = transformed[:, QI_slices[0]]
    assert ((np.sum(transformed_slice, axis=1) == 1).all())
    assert ((np.max(transformed_slice, axis=1) == 1).all())
    assert ((np.min(transformed_slice, axis=1) == 0).all())


def test_minimizer_ndarray_one_hot_single_value():
    x_train = np.array([[23, 0, 1, 0, 165],
                        [45, 0, 1, 0, 158],
                        [56, 1, 0, 0, 123],
                        [67, 0, 1, 0, 154],
                        [45, 1, 0, 0, 149],
                        [42, 1, 0, 0, 166],
                        [73, 0, 1, 0, 172],
                        [94, 0, 1, 0, 168],
                        [69, 0, 1, 0, 175],
                        [24, 1, 0, 0, 181],
                        [18, 1, 0, 0, 190]])
    y_train = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0])

    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    predictions = model.predict(x_train)

    features = ['0', '1', '2', '3', '4']
    QI = [0, 1, 2, 3]
    QI_slices = [[1, 2, 3]]
    target_accuracy = 0.7
    gen = GeneralizeToRepresentative(model, target_accuracy=target_accuracy, feature_slices=QI_slices,
                                     features_to_minimize=QI)
    gen.fit(dataset=ArrayDataset(x_train, predictions))
    transformed = gen.transform(dataset=ArrayDataset(x_train))
    gener = gen.generalizations
    expected_generalizations = {'categories': {}, 'category_representatives': {},
                                'range_representatives': {'0': [34.5]}, 'ranges': {'0': [34.5]},
                                'untouched': ['4', '1', '2', '3']}

    compare_generalizations(gener, expected_generalizations)

    check_features(features, expected_generalizations, transformed, x_train)
    ncp = gen.ncp.transform_score
    check_ncp(ncp, expected_generalizations)

    rel_accuracy = model.score(transformed, predictions)
    assert ((rel_accuracy >= target_accuracy) or (target_accuracy - rel_accuracy) <= ACCURACY_DIFF)
    transformed_slice = transformed[:, QI_slices[0]]
    assert ((np.sum(transformed_slice, axis=1) == 1).all())
    assert ((np.max(transformed_slice, axis=1) == 1).all())
    assert ((np.min(transformed_slice, axis=1) == 0).all())


def test_minimizer_ndarray_one_hot_gen():
    x_train = np.array([[23, 0, 1, 165],
                        [45, 0, 1, 158],
                        [56, 1, 0, 123],
                        [67, 0, 1, 154],
                        [45, 1, 0, 149],
                        [42, 1, 0, 166],
                        [73, 0, 1, 172],
                        [94, 0, 1, 168],
                        [69, 0, 1, 175],
                        [24, 1, 0, 181],
                        [18, 1, 0, 190]])
    y_train = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0])

    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    predictions = model.predict(x_train)

    features = ['0', '1', '2', '3']
    QI = [0, 1, 2]
    QI_slices = [[1, 2]]
    target_accuracy = 0.2
    gen = GeneralizeToRepresentative(model, target_accuracy=target_accuracy, feature_slices=QI_slices,
                                     features_to_minimize=QI)
    gen.fit(dataset=ArrayDataset(x_train, predictions))
    transformed = gen.transform(dataset=ArrayDataset(x_train))
    gener = gen.generalizations
    expected_generalizations = {'categories': {'1': [[0, 1]], '2': [[0, 1]]},
                                'category_representatives': {'1': [0], '2': [1]},
                                'range_representatives': {'0': []}, 'ranges': {'0': []}, 'untouched': ['3']}

    compare_generalizations(gener, expected_generalizations)

    check_features(features, expected_generalizations, transformed, x_train)
    ncp = gen.ncp.transform_score
    check_ncp(ncp, expected_generalizations)

    rel_accuracy = model.score(transformed, predictions)
    assert ((rel_accuracy >= target_accuracy) or (target_accuracy - rel_accuracy) <= ACCURACY_DIFF)
    transformed_slice = transformed[:, QI_slices[0]]
    assert ((np.sum(transformed_slice, axis=1) == 1).all())
    assert ((np.max(transformed_slice, axis=1) == 1).all())
    assert ((np.min(transformed_slice, axis=1) == 0).all())


def test_minimizer_ndarray_one_hot_multi():
    x_train = np.array([[23, 0, 1, 0, 0, 1, 165],
                        [45, 0, 1, 0, 0, 1, 158],
                        [56, 1, 0, 0, 0, 1, 123],
                        [67, 0, 1, 1, 0, 0, 154],
                        [45, 1, 0, 1, 0, 0, 149],
                        [42, 1, 0, 1, 0, 0, 166],
                        [73, 0, 1, 0, 0, 1, 172],
                        [94, 0, 1, 0, 1, 0, 168],
                        [69, 0, 1, 0, 1, 0, 175],
                        [24, 1, 0, 0, 1, 0, 181],
                        [18, 1, 0, 0, 0, 1, 190]])
    y_train = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0])

    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    predictions = model.predict(x_train)

    features = ['0', '1', '2', '3', '4', '5', '6']
    QI = [0, 1, 2, 3, 4, 5]
    QI_slices = [[1, 2], [3, 4, 5]]
    target_accuracy = 0.2
    gen = GeneralizeToRepresentative(model, target_accuracy=target_accuracy, feature_slices=QI_slices,
                                     features_to_minimize=QI)
    gen.fit(dataset=ArrayDataset(x_train, predictions))
    transformed = gen.transform(dataset=ArrayDataset(x_train))
    gener = gen.generalizations
    expected_generalizations = {'categories':
                                {'1': [[0, 1]], '2': [[0, 1]], '3': [[0, 1]], '4': [[0, 1]], '5': [[0, 1]]},
                                'category_representatives': {'1': [0], '2': [1], '3': [0], '4': [1], '5': [0]},
                                'range_representatives': {'0': []}, 'ranges': {'0': []}, 'untouched': ['6']}

    compare_generalizations(gener, expected_generalizations)

    check_features(features, expected_generalizations, transformed, x_train)
    ncp = gen.ncp.transform_score
    check_ncp(ncp, expected_generalizations)

    rel_accuracy = model.score(transformed, predictions)
    assert ((rel_accuracy >= target_accuracy) or (target_accuracy - rel_accuracy) <= ACCURACY_DIFF)
    transformed_slice = transformed[:, QI_slices[0]]
    assert ((np.sum(transformed_slice, axis=1) == 1).all())
    assert ((np.max(transformed_slice, axis=1) == 1).all())
    assert ((np.min(transformed_slice, axis=1) == 0).all())
    transformed_slice = transformed[:, QI_slices[1]]
    assert ((np.sum(transformed_slice, axis=1) == 1).all())
    assert ((np.max(transformed_slice, axis=1) == 1).all())
    assert ((np.min(transformed_slice, axis=1) == 0).all())


def test_minimizer_ndarray_one_hot_multi2():
    x_train = np.array([[0, 0, 1],
                        [0, 0, 1],
                        [0, 1, 0],
                        [0, 1, 0],
                        [1, 0, 0],
                        [1, 0, 0]])
    y_train = np.array([1, 1, 2, 2, 0, 0])

    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    predictions = model.predict(x_train)

    features = ['0', '1', '2']
    QI = [0, 1, 2]
    QI_slices = [[0, 1, 2]]
    target_accuracy = 0.2
    gen = GeneralizeToRepresentative(model, target_accuracy=target_accuracy, feature_slices=QI_slices,
                                     features_to_minimize=QI)
    gen.fit(dataset=ArrayDataset(x_train, predictions))
    transformed = gen.transform(dataset=ArrayDataset(x_train))
    gener = gen.generalizations
    expected_generalizations = {'categories': {'0': [[0, 1]], '1': [[0, 1]], '2': [[0, 1]]},
                                'category_representatives': {'0': [0], '1': [0], '2': [1]}, 'range_representatives': {},
                                'ranges': {}, 'untouched': []}

    compare_generalizations(gener, expected_generalizations)

    check_features(features, expected_generalizations, transformed, x_train)
    ncp = gen.ncp.transform_score
    check_ncp(ncp, expected_generalizations)

    rel_accuracy = model.score(transformed, predictions)
    assert ((rel_accuracy >= target_accuracy) or (target_accuracy - rel_accuracy) <= ACCURACY_DIFF)
    transformed_slice = transformed[:, QI_slices[0]]
    assert ((np.sum(transformed_slice, axis=1) == 1).all())
    assert ((np.max(transformed_slice, axis=1) == 1).all())
    assert ((np.min(transformed_slice, axis=1) == 0).all())


def test_anonymize_pandas_one_hot():
    features = ["age", "gender_M", "gender_F", "height"]
    x_train = np.array([[23, 0, 1, 165],
                        [45, 0, 1, 158],
                        [56, 1, 0, 123],
                        [67, 0, 1, 154],
                        [45, 1, 0, 149],
                        [42, 1, 0, 166],
                        [73, 0, 1, 172],
                        [94, 0, 1, 168],
                        [69, 0, 1, 175],
                        [24, 1, 0, 181],
                        [18, 1, 0, 190]])
    y_train = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
    x_train = pd.DataFrame(x_train, columns=features)
    y_train = pd.Series(y_train)

    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    predictions = model.predict(x_train)

    QI = ["age", "gender_M", "gender_F"]
    QI_slices = [["gender_M", "gender_F"]]
    target_accuracy = 0.7
    gen = GeneralizeToRepresentative(model, target_accuracy=target_accuracy, feature_slices=QI_slices,
                                     features_to_minimize=QI)
    gen.fit(dataset=ArrayDataset(x_train, predictions))
    transformed = gen.transform(dataset=ArrayDataset(x_train))
    gener = gen.generalizations
    expected_generalizations = {'categories': {}, 'category_representatives': {},
                                'range_representatives': {'age': [34.5]},
                                'ranges': {'age': [34.5]}, 'untouched': ['height', 'gender_M', 'gender_F']}

    compare_generalizations(gener, expected_generalizations)

    check_features(features, expected_generalizations, transformed, x_train, True)
    ncp = gen.ncp.transform_score
    check_ncp(ncp, expected_generalizations)

    rel_accuracy = model.score(transformed, predictions)
    assert ((rel_accuracy >= target_accuracy) or (target_accuracy - rel_accuracy) <= ACCURACY_DIFF)
    transformed_slice = transformed.loc[:, QI_slices[0]]
    assert ((np.sum(transformed_slice, axis=1) == 1).all())
    assert ((np.max(transformed_slice, axis=1) == 1).all())
    assert ((np.min(transformed_slice, axis=1) == 0).all())


def test_keras_model():
    (x, y), (x_test, y_test) = get_iris_dataset_np()

    base_est = Sequential()
    base_est.add(Input(shape=(4,)))
    base_est.add(Dense(10, activation="relu"))
    base_est.add(Dense(3, activation='softmax'))

    base_est.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model = KerasClassifier(base_est, CLASSIFIER_SINGLE_OUTPUT_CLASS_PROBABILITIES)
    model.fit(ArrayDataset(x, y))
    ad = ArrayDataset(x_test)
    predictions = model.predict(ad)
    if predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)
    target_accuracy = 0.5
    gen = GeneralizeToRepresentative(model, target_accuracy=target_accuracy)
    test_dataset = ArrayDataset(x_test, predictions)

    gen.fit(dataset=test_dataset)
    transformed = gen.transform(dataset=ad)
    gener = gen.generalizations

    features = ['0', '1', '2', '3']
    check_features(features, gener, transformed, x)
    ncp = gen.ncp.transform_score
    check_ncp(ncp, gener)

    rel_accuracy = model.score(ArrayDataset(transformed, predictions))
    assert ((rel_accuracy >= target_accuracy) or (target_accuracy - rel_accuracy) <= ACCURACY_DIFF)


class PytorchModel(nn.Module):

    def __init__(self, num_classes, num_features):
        super(PytorchModel, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.Tanh(), )

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Tanh(), )

        self.fc3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh(), )

        self.fc4 = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        return self.classifier(out)


def test_minimizer_pytorch(data_three_features):
    x, y, features = data_three_features
    x = x.astype(np.float32)
    qi = ['age', 'weight']

    from apt.utils.datasets.datasets import PytorchData
    from apt.utils.models.pytorch_model import PyTorchClassifier

    base_est = PytorchModel(2, 3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(base_est.parameters(), lr=0.01)

    model = PyTorchClassifier(model=base_est,
                              output_type=CLASSIFIER_SINGLE_OUTPUT_CLASS_LOGITS,
                              loss=criterion,
                              optimizer=optimizer,
                              input_shape=(3,),
                              nb_classes=2)
    model.fit(PytorchData(x, y), save_entire_model=False, nb_epochs=10)

    ad = ArrayDataset(x)
    predictions = model.predict(ad)
    if predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)
    target_accuracy = 0.5
    gen = GeneralizeToRepresentative(model, target_accuracy=target_accuracy, features_to_minimize=qi)
    train_dataset = ArrayDataset(x, predictions, features_names=features)
    gen.fit(dataset=train_dataset)
    transformed = gen.transform(dataset=ad)
    gener = gen.generalizations
    expected_generalizations = {'ranges': {'age': [], 'weight': []}, 'categories': {}, 'untouched': ['height']}
    compare_generalizations(gener, expected_generalizations)
    check_features(features, expected_generalizations, transformed, x)
    assert ((np.delete(transformed, [0, 2], axis=1) == np.delete(x, [0, 2], axis=1)).all())
    ncp = gen.ncp.transform_score
    check_ncp(ncp, expected_generalizations)

    rel_accuracy = model.score(ArrayDataset(transformed.astype(np.float32), predictions))
    assert ((rel_accuracy >= target_accuracy) or (target_accuracy - rel_accuracy) <= ACCURACY_DIFF)


def test_minimizer_pytorch_iris():
    features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    (x_train, y_train), _ = get_iris_dataset_np()
    x_train = x_train.astype(np.float32)
    qi = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

    from apt.utils.datasets.datasets import PytorchData
    from apt.utils.models.pytorch_model import PyTorchClassifier

    base_est = PytorchModel(3, 4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(base_est.parameters(), lr=0.01)

    model = PyTorchClassifier(model=base_est,
                              output_type=CLASSIFIER_SINGLE_OUTPUT_CLASS_LOGITS,
                              loss=criterion,
                              optimizer=optimizer,
                              input_shape=(4,),
                              nb_classes=3)
    model.fit(PytorchData(x_train, y_train), save_entire_model=False, nb_epochs=10)

    predictions = model.predict(ArrayDataset(x_train))
    if predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)
    target_accuracy = 0.99
    gen = GeneralizeToRepresentative(model, target_accuracy=target_accuracy, features_to_minimize=qi)
    transformed = gen.fit_transform(dataset=ArrayDataset(x_train, predictions, features_names=features))
    gener = gen.generalizations

    check_features(features, gener, transformed, x_train)
    ncp = gen.ncp.transform_score
    check_ncp(ncp, gener)

    rel_accuracy = model.score(ArrayDataset(transformed.astype(np.float32), predictions))
    assert ((rel_accuracy >= target_accuracy) or (target_accuracy - rel_accuracy) <= ACCURACY_DIFF)


def test_minimizer_pytorch_multi_label_binary():
    class multi_label_binary_model(nn.Module):
        def __init__(self, num_labels, num_features):
            super(multi_label_binary_model, self).__init__()

            self.fc1 = nn.Sequential(
                nn.Linear(num_features, 256),
                nn.Tanh(), )

            self.classifier1 = nn.Linear(256, num_labels)

        def forward(self, x):
            return self.classifier1(self.fc1(x))
            # missing sigmoid on each output

    class FocalLoss(nn.Module):
        def __init__(self, gamma=2, alpha=0.5):
            super(FocalLoss, self).__init__()
            self.gamma = gamma
            self.alpha = alpha

        def forward(self, input, target):
            bce_loss = functional.binary_cross_entropy_with_logits(input, target, reduction='none')

            p = sigmoid(input)
            p = where(target >= 0.5, p, 1 - p)

            modulating_factor = (1 - p) ** self.gamma
            alpha = self.alpha * target + (1 - self.alpha) * (1 - target)
            focal_loss = alpha * modulating_factor * bce_loss

            return focal_loss.mean()

    (x_train, y_train), _ = get_iris_dataset_np()
    features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    qi = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

    # make multi-label binary
    y_train = np.column_stack((y_train, y_train, y_train))
    y_train[y_train > 1] = 1
    x_train = x_train.astype(np.float32)
    y_train = y_train.astype(np.float32)

    orig_model = multi_label_binary_model(3, 4)
    criterion = FocalLoss()
    optimizer = optim.RMSprop(orig_model.parameters(), lr=0.01)

    model = PyTorchClassifier(model=orig_model,
                              output_type=CLASSIFIER_MULTI_OUTPUT_BINARY_LOGITS,
                              loss=criterion,
                              optimizer=optimizer,
                              input_shape=(24,),
                              nb_classes=3)
    model.fit(PytorchData(x_train, y_train), save_entire_model=False, nb_epochs=10)
    predictions = model.predict(PytorchData(x_train, y_train))
    predictions = expit(predictions)
    predictions[predictions < 0.5] = 0
    predictions[predictions >= 0.5] = 1

    target_accuracy = 0.99
    gen = GeneralizeToRepresentative(model, target_accuracy=target_accuracy, features_to_minimize=qi)
    transformed = gen.fit_transform(dataset=ArrayDataset(x_train, predictions, features_names=features))
    gener = gen.generalizations

    check_features(features, gener, transformed, x_train)
    ncp = gen.ncp.transform_score
    check_ncp(ncp, gener)

    rel_accuracy = model.score(ArrayDataset(transformed.astype(np.float32), predictions))
    assert ((rel_accuracy >= target_accuracy) or (target_accuracy - rel_accuracy) <= ACCURACY_DIFF)


def test_untouched():
    cells = [{"id": 1, "ranges": {"age": {"start": None, "end": 38}}, "label": 0,
              'categories': {'gender': ['male']}, "representative": {"age": 26, "height": 149}},
             {"id": 2, "ranges": {"age": {"start": 39, "end": None}}, "label": 1,
              'categories': {'gender': ['female']}, "representative": {"age": 58, "height": 163}},
             {"id": 3, "ranges": {"age": {"start": None, "end": 38}}, "label": 0,
              'categories': {'gender': ['male']}, "representative": {"age": 31, "height": 184}},
             {"id": 4, "ranges": {"age": {"start": 39, "end": None}}, "label": 1,
              'categories': {'gender': ['male', 'female']}, "representative": {"age": 45, "height": 176}}
             ]
    gen = GeneralizeToRepresentative(cells=cells)
    gen._calculate_generalizations()
    gener = gen.generalizations
    expected_generalizations = {'ranges': {'age': [38, 39]}, 'categories': {}, 'untouched': ['gender']}
    compare_generalizations(gener, expected_generalizations)


def test_errors():
    features = ['age', 'height']
    X = np.array([[23, 165],
                  [45, 158],
                  [56, 123],
                  [67, 154],
                  [45, 149],
                  [42, 166],
                  [73, 172],
                  [94, 168],
                  [69, 175],
                  [24, 181],
                  [18, 190]])
    y = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
    base_est = DecisionTreeClassifier(random_state=0, min_samples_split=2,
                                      min_samples_leaf=1)
    model = SklearnClassifier(base_est, CLASSIFIER_SINGLE_OUTPUT_CLASS_PROBABILITIES)
    model.fit(ArrayDataset(X, y))
    ad = ArrayDataset(X)
    predictions = model.predict(ad)
    if predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)
    gen = GeneralizeToRepresentative(model, generalize_using_transform=False)
    train_dataset = ArrayDataset(X, predictions, features_names=features)
    gen.fit(dataset=train_dataset)
    with pytest.raises(ValueError):
        gen.transform(X)
