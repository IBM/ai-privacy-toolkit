import pytest
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from sklearn.datasets import load_boston
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from apt.minimization import GeneralizeToRepresentative
from sklearn.tree import DecisionTreeClassifier
from apt.utils import get_iris_dataset, get_adult_dataset, get_nursery_dataset


@pytest.fixture
def data():
    return load_boston(return_X_y=True)


def test_minimizer_params(data):
    # Assume two features, age and height, and boolean label
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
    X = np.array([[23, 165],
                  [45, 158],
                  [18, 190]])
    y = [1, 1, 0]
    base_est = DecisionTreeClassifier(random_state=0, min_samples_split=2,
                                      min_samples_leaf=1)
    base_est.fit(X, y)

    gen = GeneralizeToRepresentative(base_est, features=features, cells=cells)
    gen.fit()
    transformed = gen.transform(X)
    expected_transformed = np.array([[26, 149],
                                    [58, 163],
                                    [31, 184]])
    assert(np.array_equal(expected_transformed, transformed))


def test_minimizer_fit(data):
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
    print(X)
    y = [1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0]
    base_est = DecisionTreeClassifier(random_state=0, min_samples_split=2,
                                      min_samples_leaf=1)
    base_est.fit(X, y)
    predictions = base_est.predict(X)

    gen = GeneralizeToRepresentative(base_est, features=features, target_accuracy=0.5)
    gen.fit(X, predictions)
    transformed = gen.transform(X)
    gener = gen.generalizations_
    expexted_generalizations = {'ranges': {}, 'categories': {}, 'untouched': ['age', 'height']}
    for key in expexted_generalizations['ranges']:
        assert (set(expexted_generalizations['ranges'][key]) == set(gener['ranges'][key]))
    for key in expexted_generalizations['categories']:
        assert (set([frozenset(sl) for sl in expexted_generalizations['categories'][key]]) ==
                set([frozenset(sl) for sl in gener['categories'][key]]))
    assert (set(expexted_generalizations['untouched']) == set(gener['untouched']))
    modified_features = [f for f in features if
                         f in expexted_generalizations['categories'].keys() or f in expexted_generalizations[
                             'ranges'].keys()]
    indexes = []
    for i in range(len(features)):
        if features[i] in modified_features:
            indexes.append(i)
    assert ((np.delete(transformed, indexes, axis=1) == np.delete(X, indexes, axis=1)).all())
    ncp = gen.ncp_
    if len(expexted_generalizations['ranges'].keys()) > 0 or len(expexted_generalizations['categories'].keys()) > 0:
        assert (ncp > 0)
        assert (((transformed[indexes]) != (X[indexes])).any())


def test_minimizer_fit_pandas(data):
    features = ['age', 'height', 'sex', 'ola']
    X = [[23, 165, 'f', 'aa'],
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
    y = [1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0]
    X = pd.DataFrame(X, columns=features)

    numeric_features = ["age", "height"]
    numeric_transformer = Pipeline(
        steps=[('imputer', SimpleImputer(strategy='constant', fill_value=0))]
    )

    categorical_features = ["sex", "ola"]
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    encoded = preprocessor.fit_transform(X)
    base_est = DecisionTreeClassifier(random_state=0, min_samples_split=2,
                                      min_samples_leaf=1)
    base_est.fit(encoded, y)
    predictions = base_est.predict(encoded)
    # Append classifier to preprocessing pipeline.
    # Now we have a full prediction pipeline.
    gen = GeneralizeToRepresentative(base_est, features=features, target_accuracy=0.5,
                                     categorical_features=categorical_features)
    gen.fit(X, predictions)
    transformed = gen.transform(X)
    gener = gen.generalizations_
    expexted_generalizations = {'ranges': {'age': []}, 'categories': {}, 'untouched': ['sex', 'height', 'ola']}
    for key in expexted_generalizations['ranges']:
        assert (set(expexted_generalizations['ranges'][key]) == set(gener['ranges'][key]))
    for key in expexted_generalizations['categories']:
        assert (set([frozenset(sl) for sl in expexted_generalizations['categories'][key]]) ==
                set([frozenset(sl) for sl in gener['categories'][key]]))
    assert (set(expexted_generalizations['untouched']) == set(gener['untouched']))
    modified_features = [f for f in features if
                         f in expexted_generalizations['categories'].keys() or f in expexted_generalizations[
                             'ranges'].keys()]
    assert (transformed.drop(modified_features, axis=1).equals(X.drop(modified_features, axis=1)))
    ncp = gen.ncp_
    if len(expexted_generalizations['ranges'].keys()) > 0 or len(expexted_generalizations['categories'].keys()) > 0:
        assert (ncp > 0)
        assert (((transformed[modified_features]).equals(X[modified_features])) == False)


def test_minimizer_params_categorical(data):
    # Assume three features, age, sex and height, and boolean label
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
    X = [[23, 165, 'f'],
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

    y = [1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0]
    X = pd.DataFrame(X, columns=features)
    numeric_features = ["age", "height"]
    numeric_transformer = Pipeline(
        steps=[('imputer', SimpleImputer(strategy='constant', fill_value=0))]
    )

    categorical_features = ["sex"]
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    encoded = preprocessor.fit_transform(X)
    base_est = DecisionTreeClassifier(random_state=0, min_samples_split=2,
                                      min_samples_leaf=1)
    base_est.fit(encoded, y)
    predictions = base_est.predict(encoded)
    # Append classifier to preprocessing pipeline.
    # Now we have a full prediction pipeline.
    gen = GeneralizeToRepresentative(base_est, features=features, target_accuracy=0.5,
                                     categorical_features=categorical_features)
    gen.fit(X, predictions)
    transformed = gen.transform(X)
    gener = gen.generalizations_
    expexted_generalizations = {'ranges': {'age': []}, 'categories': {}, 'untouched': ['height', 'sex']}
    for key in expexted_generalizations['ranges']:
        assert (set(expexted_generalizations['ranges'][key]) == set(gener['ranges'][key]))
    for key in expexted_generalizations['categories']:
        assert (set([frozenset(sl) for sl in expexted_generalizations['categories'][key]]) ==
                set([frozenset(sl) for sl in gener['categories'][key]]))
    assert (set(expexted_generalizations['untouched']) == set(gener['untouched']))
    modified_features = [f for f in features if
                         f in expexted_generalizations['categories'].keys() or f in expexted_generalizations[
                             'ranges'].keys()]
    assert (transformed.drop(modified_features, axis=1).equals(X.drop(modified_features, axis=1)))
    ncp = gen.ncp_
    if len(expexted_generalizations['ranges'].keys()) > 0 or len(expexted_generalizations['categories'].keys()) > 0:
        assert (ncp > 0)
        assert (((transformed[modified_features]).equals(X[modified_features])) == False)


def test_minimize_ndarray_iris():
    features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    (x_train, y_train), _ = get_iris_dataset()
    model = DecisionTreeClassifier(random_state=0, min_samples_split=2,
                                   min_samples_leaf=1)
    model.fit(x_train, y_train)
    pred = model.predict(x_train)

    gen = GeneralizeToRepresentative(model, target_accuracy=0.7, features=features)
    gen.fit(x_train, pred)
    transformed = gen.transform(x_train)
    gener = gen.generalizations_
    expexted_generalizations = {
        'ranges': {'sepal length (cm)': [5.0], 'sepal width (cm)': [], 'petal length (cm)': [4.950000047683716],
                   'petal width (cm)': [0.800000011920929, 1.699999988079071]}, 'categories': {}, 'untouched': []}
    for key in expexted_generalizations['ranges']:
        assert (set(expexted_generalizations['ranges'][key]) == set(gener['ranges'][key]))
    for key in expexted_generalizations['categories']:
        assert (set([frozenset(sl) for sl in expexted_generalizations['categories'][key]]) ==
                set([frozenset(sl) for sl in gener['categories'][key]]))
    assert (set(expexted_generalizations['untouched']) == set(gener['untouched']))
    modified_features = [f for f in features if
                         f in expexted_generalizations['categories'].keys() or f in expexted_generalizations[
                             'ranges'].keys()]
    indexes = []
    for i in range(len(features)):
        if features[i] in modified_features:
            indexes.append(i)
    assert ((np.delete(transformed, indexes, axis=1) == np.delete(x_train, indexes, axis=1)).all())
    ncp = gen.ncp_
    if len(expexted_generalizations['ranges'].keys()) > 0 or len(expexted_generalizations['categories'].keys()) > 0:
        assert (ncp > 0)
        assert (((transformed[indexes]) != (x_train[indexes])).any())


def test_minimize_pandas_nursery():
    (x_train, y_train), _ = get_nursery_dataset()
    x_train = x_train.astype(str)
    x_train.reset_index(inplace=True, drop=True)
    y_train.reset_index(inplace=True, drop=True)
    QI = ["finance", "social", "health"]
    features = ["parents", "has_nurs", "form", "children", "housing", "finance", "social", "health"]
    categorical_features = ["parents", "has_nurs", "form", "housing", "finance", "social", "health", 'children']
    numeric_features = [f for f in features if f not in categorical_features]
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
    encoded = preprocessor.fit_transform(x_train)
    base_est = DecisionTreeClassifier(random_state=0, min_samples_split=2,
                                      min_samples_leaf=1)
    base_est.fit(encoded, y_train)
    predictions = base_est.predict(encoded)

    gen = GeneralizeToRepresentative(base_est, target_accuracy=0.8, features=features,
                                     categorical_features=categorical_features)
    gen.fit(x_train, predictions)
    transformed = gen.transform(x_train)
    gener = gen.generalizations_
    expexted_generalizations = {'ranges': {}, 'categories': {'parents': [['great_pret', 'pretentious', 'usual']],
                                                             'has_nurs': [['critical', 'less_proper', 'proper'],
                                                                          ['very_crit'], ['improper']], 'form': [
            ['foster', 'completed', 'complete', 'incomplete']], 'housing': [['convenient', 'less_conv', 'critical']],
                                                             'finance': [['convenient', 'inconv']],
                                                             'social': [['problematic', 'nonprob', 'slightly_prob']],
                                                             'health': [['priority'], ['recommended'], ['not_recom']],
                                                             'children': [['2', '3', '4', '1']]}, 'untouched': []}
    for key in expexted_generalizations['ranges']:
        assert (set(expexted_generalizations['ranges'][key]) == set(gener['ranges'][key]))
    for key in expexted_generalizations['categories']:
        assert (set([frozenset(sl) for sl in expexted_generalizations['categories'][key]]) ==
                set([frozenset(sl) for sl in gener['categories'][key]]))
    assert (set(expexted_generalizations['untouched']) == set(gener['untouched']))
    modified_features = [f for f in features if
                         f in expexted_generalizations['categories'].keys() or f in expexted_generalizations[
                             'ranges'].keys()]
    assert (transformed.drop(modified_features, axis=1).equals(x_train.drop(modified_features, axis=1)))
    ncp = gen.ncp_
    if len(expexted_generalizations['ranges'].keys()) > 0 or len(expexted_generalizations['categories'].keys()) > 0:
        assert (ncp > 0)
        assert (((transformed[modified_features]).equals(x_train[modified_features])) == False)


def test_minimize_pandas_adult():
    (x_train, y_train), _ = get_adult_dataset()
    x_train = x_train.head(5000)
    y_train = y_train.head(5000)

    features = ['age', 'workclass', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']

    categorical_features = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                            'native-country']

    numeric_features = [f for f in features if f not in categorical_features]
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
    encoded = preprocessor.fit_transform(x_train)
    base_est = DecisionTreeClassifier(random_state=0, min_samples_split=2,
                                      min_samples_leaf=1)
    base_est.fit(encoded, y_train)
    predictions = base_est.predict(encoded)

    gen = GeneralizeToRepresentative(base_est, target_accuracy=0.8, features=features,
                                     categorical_features=categorical_features)
    gen.fit(x_train, predictions)
    transformed = gen.transform(x_train)
    gener = gen.generalizations_
    expexted_generalizations = {
        'ranges': {'age': [20.0], 'education-num': [11.5, 12.5], 'capital-gain': [5095.5, 7139.5], 'capital-loss': [],
                   'hours-per-week': []}, 'categories': {'workclass': [
            ['Private', 'Without-pay', 'Self-emp-not-inc', '?', 'Federal-gov', 'Self-emp-inc', 'State-gov',
             'Local-gov']], 'marital-status': [
            ['Married-civ-spouse', 'Never-married', 'Widowed', 'Married-AF-spouse', 'Separated',
             'Married-spouse-absent'], ['Divorced']], 'occupation': [
            ['Transport-moving', 'Priv-house-serv', '?', 'Armed-Forces', 'Prof-specialty', 'Farming-fishing',
             'Exec-managerial', 'Machine-op-inspct', 'Other-service', 'Sales', 'Protective-serv', 'Handlers-cleaners',
             'Tech-support', 'Craft-repair', 'Adm-clerical']], 'relationship': [
            ['Not-in-family', 'Own-child', 'Wife', 'Other-relative', 'Husband', 'Unmarried']], 'race': [
            ['Other', 'Asian-Pac-Islander', 'Black', 'White', 'Amer-Indian-Eskimo']], 'sex': [['Male', 'Female']],
            'native-country': [
                ['LatinAmerica', 'Other', 'UnitedStates', 'SouthAmerica',
                 'BritishCommonwealth', 'Euro_2', 'Unknown', 'China',
                 'Euro_1', 'SE_Asia']]}, 'untouched': []}
    for key in expexted_generalizations['ranges']:
        assert (set(expexted_generalizations['ranges'][key]) == set(gener['ranges'][key]))
    for key in expexted_generalizations['categories']:
        assert (set([frozenset(sl) for sl in expexted_generalizations['categories'][key]]) ==
                set([frozenset(sl) for sl in gener['categories'][key]]))
    assert (set(expexted_generalizations['untouched']) == set(gener['untouched']))
    modified_features = [f for f in features if
                         f in expexted_generalizations['categories'].keys() or f in expexted_generalizations[
                             'ranges'].keys()]
    assert (transformed.drop(modified_features, axis=1).equals(x_train.drop(modified_features, axis=1)))
    ncp = gen.ncp_
    if len(expexted_generalizations['ranges'].keys()) > 0 or len(expexted_generalizations['categories'].keys()) > 0:
        assert (ncp > 0)
        assert (((transformed[modified_features]).equals(x_train[modified_features])) == False)
