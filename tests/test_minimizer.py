import pytest
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer

from sklearn.datasets import load_boston, load_diabetes
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

from apt.minimization import GeneralizeToRepresentative
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from apt.utils.dataset_utils import get_iris_dataset_np, get_adult_dataset_pd, get_german_credit_dataset_pd
from apt.utils.datasets import ArrayDataset
from apt.utils.models import SklearnClassifier, ModelOutputType, SklearnRegressor, KerasClassifier

tf.compat.v1.disable_eager_execution()


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
    model = SklearnClassifier(base_est, ModelOutputType.CLASSIFIER_PROBABILITIES)
    model.fit(ArrayDataset(X, y))

    gen = GeneralizeToRepresentative(model, cells=cells)
    gen.fit()
    gen.transform(dataset=ArrayDataset(X, features_names=features))


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
    y = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
    base_est = DecisionTreeClassifier(random_state=0, min_samples_split=2,
                                      min_samples_leaf=1)
    model = SklearnClassifier(base_est, ModelOutputType.CLASSIFIER_PROBABILITIES)
    model.fit(ArrayDataset(X, y))
    ad = ArrayDataset(X)
    predictions = model.predict(ad)
    if predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)
    target_accuracy = 0.5
    gen = GeneralizeToRepresentative(model, target_accuracy=target_accuracy)
    train_dataset = ArrayDataset(X, predictions, features_names=features)

    gen.fit(dataset=train_dataset)
    transformed = gen.transform(dataset=ad)
    gener = gen.generalizations
    expected_generalizations = {'ranges': {}, 'categories': {}, 'untouched': ['height', 'age']}

    for key in expected_generalizations['ranges']:
        assert (set(expected_generalizations['ranges'][key]) == set(gener['ranges'][key]))
    for key in expected_generalizations['categories']:
        assert (set([frozenset(sl) for sl in expected_generalizations['categories'][key]])
                == set([frozenset(sl) for sl in gener['categories'][key]]))
    assert (set(expected_generalizations['untouched']) == set(gener['untouched']))
    modified_features = [f for f in features if
                         f in expected_generalizations['categories'].keys() or f in expected_generalizations[
                             'ranges'].keys()]
    indexes = []
    for i in range(len(features)):
        if features[i] in modified_features:
            indexes.append(i)
    assert ((np.delete(transformed, indexes, axis=1) == np.delete(X, indexes, axis=1)).all())
    ncp = gen.ncp
    if len(expected_generalizations['ranges'].keys()) > 0 or len(expected_generalizations['categories'].keys()) > 0:
        assert (ncp > 0)
        assert (((transformed[indexes]) != (X[indexes])).any())

    rel_accuracy = model.score(ArrayDataset(transformed, predictions))
    assert ((rel_accuracy >= target_accuracy) or (target_accuracy - rel_accuracy) <= 0.05)


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
    y = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
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
    encoded = pd.DataFrame(encoded)
    base_est = DecisionTreeClassifier(random_state=0, min_samples_split=2,
                                      min_samples_leaf=1)
    model = SklearnClassifier(base_est, ModelOutputType.CLASSIFIER_PROBABILITIES)
    model.fit(ArrayDataset(encoded, y))
    predictions = model.predict(ArrayDataset(encoded))
    if predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)

    # Append classifier to preprocessing pipeline.
    # Now we have a full prediction pipeline.
    target_accuracy = 0.5
    gen = GeneralizeToRepresentative(model, target_accuracy=target_accuracy,
                                     categorical_features=categorical_features)
    train_dataset = ArrayDataset(X, predictions)
    gen.fit(dataset=train_dataset)
    transformed = gen.transform(dataset=ArrayDataset(X))
    gener = gen.generalizations
    expected_generalizations = {'ranges': {'age': []}, 'categories': {'sex': [['f', 'm']], 'ola': [['aa', 'bb']]},
                                'untouched': ['height']}

    for key in expected_generalizations['ranges']:
        assert (set(expected_generalizations['ranges'][key]) == set(gener['ranges'][key]))
    for key in expected_generalizations['categories']:
        assert (set([frozenset(sl) for sl in expected_generalizations['categories'][key]])
                == set([frozenset(sl) for sl in gener['categories'][key]]))
    assert (set(expected_generalizations['untouched']) == set(gener['untouched']))
    modified_features = [f for f in features if
                         f in expected_generalizations['categories'].keys() or f in expected_generalizations[
                             'ranges'].keys()]
    np.testing.assert_array_equal(transformed.drop(modified_features, axis=1), X.drop(modified_features, axis=1))
    ncp = gen.ncp
    if len(expected_generalizations['ranges'].keys()) > 0 or len(expected_generalizations['categories'].keys()) > 0:
        assert (ncp > 0)
        assert (((transformed[modified_features]).equals(X[modified_features])) is False)

    rel_accuracy = model.score(ArrayDataset(preprocessor.transform(transformed), predictions))
    assert ((rel_accuracy >= target_accuracy) or (target_accuracy - rel_accuracy) <= 0.05)


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

    y = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
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
    encoded = pd.DataFrame(encoded)
    base_est = DecisionTreeClassifier(random_state=0, min_samples_split=2,
                                      min_samples_leaf=1)
    model = SklearnClassifier(base_est, ModelOutputType.CLASSIFIER_PROBABILITIES)
    model.fit(ArrayDataset(encoded, y))
    predictions = model.predict(ArrayDataset(encoded))
    if predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)
    # Append classifier to preprocessing pipeline.
    # Now we have a full prediction pipeline.
    target_accuracy = 0.5
    gen = GeneralizeToRepresentative(model, target_accuracy=target_accuracy,
                                     categorical_features=categorical_features, cells=cells)
    train_dataset = ArrayDataset(X, predictions)
    gen.fit(dataset=train_dataset)
    transformed = gen.transform(dataset=ArrayDataset(X))

    rel_accuracy = model.score(ArrayDataset(preprocessor.transform(transformed), predictions))
    assert ((rel_accuracy >= target_accuracy) or (target_accuracy - rel_accuracy) <= 0.05)


def test_minimizer_fit_QI(data):
    features = ['age', 'height', 'weight']
    X = np.array([[23, 165, 70],
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
    print(X)
    y = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
    QI = ['age', 'weight']
    base_est = DecisionTreeClassifier(random_state=0, min_samples_split=2,
                                      min_samples_leaf=1)
    model = SklearnClassifier(base_est, ModelOutputType.CLASSIFIER_PROBABILITIES)
    model.fit(ArrayDataset(X, y))
    ad = ArrayDataset(X)
    predictions = model.predict(ad)
    if predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)
    target_accuracy = 0.5
    gen = GeneralizeToRepresentative(model, target_accuracy=target_accuracy, features_to_minimize=QI)
    train_dataset = ArrayDataset(X, predictions, features_names=features)
    gen.fit(dataset=train_dataset)
    transformed = gen.transform(dataset=ad)
    gener = gen.generalizations
    expected_generalizations = {'ranges': {'age': [], 'weight': [67.5]}, 'categories': {}, 'untouched': ['height']}
    for key in expected_generalizations['ranges']:
        assert (set(expected_generalizations['ranges'][key]) == set(gener['ranges'][key]))
    for key in expected_generalizations['categories']:
        assert (set([frozenset(sl) for sl in expected_generalizations['categories'][key]])
                == set([frozenset(sl) for sl in gener['categories'][key]]))
    assert (set(expected_generalizations['untouched']) == set(gener['untouched']))
    assert ((np.delete(transformed, [0, 2], axis=1) == np.delete(X, [0, 2], axis=1)).all())
    modified_features = [f for f in features if
                         f in expected_generalizations['categories'].keys() or f in expected_generalizations[
                             'ranges'].keys()]
    indexes = []
    for i in range(len(features)):
        if features[i] in modified_features:
            indexes.append(i)
    assert ((np.delete(transformed, indexes, axis=1) == np.delete(X, indexes, axis=1)).all())
    ncp = gen.ncp
    if len(expected_generalizations['ranges'].keys()) > 0 or len(expected_generalizations['categories'].keys()) > 0:
        assert (ncp > 0)
        assert (((transformed[indexes]) != (X[indexes])).any())

    rel_accuracy = model.score(ArrayDataset(transformed, predictions))
    assert ((rel_accuracy >= target_accuracy) or (target_accuracy - rel_accuracy) <= 0.05)


def test_minimizer_fit_pandas_QI(data):
    features = ['age', 'height', 'weight', 'sex', 'ola']
    X = [[23, 165, 65, 'f', 'aa'],
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
    X = pd.DataFrame(X, columns=features)
    QI = ['age', 'weight', 'ola']

    numeric_features = ["age", "height", "weight"]
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
    encoded = pd.DataFrame(encoded)
    base_est = DecisionTreeClassifier(random_state=0, min_samples_split=2,
                                      min_samples_leaf=1)
    model = SklearnClassifier(base_est, ModelOutputType.CLASSIFIER_PROBABILITIES)
    model.fit(ArrayDataset(encoded, y))
    predictions = model.predict(ArrayDataset(encoded))
    if predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)

    # Append classifier to preprocessing pipeline.
    # Now we have a full prediction pipeline.
    target_accuracy = 0.5
    gen = GeneralizeToRepresentative(model, target_accuracy=target_accuracy,
                                     categorical_features=categorical_features, features_to_minimize=QI)
    train_dataset = ArrayDataset(X, predictions)
    gen.fit(dataset=train_dataset)
    transformed = gen.transform(dataset=ArrayDataset(X))
    gener = gen.generalizations
    expected_generalizations = {'ranges': {'age': [], 'weight': [47.0]}, 'categories': {'ola': [['bb', 'aa']]},
                                'untouched': ['height', 'sex']}

    for key in expected_generalizations['ranges']:
        assert (set(expected_generalizations['ranges'][key]) == set(gener['ranges'][key]))
    for key in expected_generalizations['categories']:
        assert (set([frozenset(sl) for sl in expected_generalizations['categories'][key]])
                == set([frozenset(sl) for sl in gener['categories'][key]]))
    assert (set(expected_generalizations['untouched']) == set(gener['untouched']))
    # assert (transformed.drop(QI, axis=1).equals(X.drop(QI, axis=1)))
    np.testing.assert_array_equal(transformed.drop(QI, axis=1), X.drop(QI, axis=1))
    modified_features = [f for f in features if
                         f in expected_generalizations['categories'].keys() or f in expected_generalizations[
                             'ranges'].keys()]
    # assert (transformed.drop(modified_features, axis=1).equals(X.drop(modified_features, axis=1)))
    np.testing.assert_array_equal(transformed.drop(modified_features, axis=1), X.drop(modified_features, axis=1))
    ncp = gen.ncp
    if len(expected_generalizations['ranges'].keys()) > 0 or len(expected_generalizations['categories'].keys()) > 0:
        assert (ncp > 0)
        assert (((transformed[modified_features]).equals(X[modified_features])) is False)

    rel_accuracy = model.score(ArrayDataset(preprocessor.transform(transformed), predictions))
    assert ((rel_accuracy >= target_accuracy) or (target_accuracy - rel_accuracy) <= 0.05)


def test_minimize_ndarray_iris():
    features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    (x_train, y_train), (x_test, y_test) = get_iris_dataset_np()
    QI = ['sepal length (cm)', 'petal length (cm)']
    base_est = DecisionTreeClassifier(random_state=0, min_samples_split=2,
                                      min_samples_leaf=1)
    model = SklearnClassifier(base_est, ModelOutputType.CLASSIFIER_PROBABILITIES)
    model.fit(ArrayDataset(x_train, y_train))
    predictions = model.predict(ArrayDataset(x_train))
    if predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)
    target_accuracy = 0.3
    gen = GeneralizeToRepresentative(model, target_accuracy=target_accuracy, features_to_minimize=QI)
    # gen.fit(dataset=ArrayDataset(x_train, predictions))
    transformed = gen.fit_transform(dataset=ArrayDataset(x_train, predictions, features_names=features))
    gener = gen.generalizations
    expected_generalizations = {'ranges': {'sepal length (cm)': [], 'petal length (cm)': [2.449999988079071]},
                                'categories': {}, 'untouched': ['petal width (cm)', 'sepal width (cm)']}

    for key in expected_generalizations['ranges']:
        assert (set(expected_generalizations['ranges'][key]) == set(gener['ranges'][key]))
    for key in expected_generalizations['categories']:
        assert (set([frozenset(sl) for sl in expected_generalizations['categories'][key]])
                == set([frozenset(sl) for sl in gener['categories'][key]]))
    assert (set(expected_generalizations['untouched']) == set(gener['untouched']))
    assert ((np.delete(transformed, [0, 2], axis=1) == np.delete(x_train, [0, 2], axis=1)).all())

    modified_features = [f for f in features if
                         f in expected_generalizations['categories'].keys() or f in expected_generalizations[
                             'ranges'].keys()]
    indexes = []
    for i in range(len(features)):
        if features[i] in modified_features:
            indexes.append(i)
    assert ((np.delete(transformed, indexes, axis=1) == np.delete(x_train, indexes, axis=1)).all())
    ncp = gen.ncp
    if len(expected_generalizations['ranges'].keys()) > 0 or len(expected_generalizations['categories'].keys()) > 0:
        assert (ncp > 0)
        assert (((transformed[indexes]) != (x_train[indexes])).any())

    rel_accuracy = model.score(ArrayDataset(transformed, predictions))
    assert ((rel_accuracy >= target_accuracy) or (target_accuracy - rel_accuracy) <= 0.05)


def test_minimize_pandas_adult():
    (x_train, y_train), (x_test, y_test) = get_adult_dataset_pd()
    x_train = x_train.head(1000)
    y_train = y_train.head(1000)

    features = ['age', 'workclass', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
    x_train = pd.DataFrame(x_train, columns=features)

    categorical_features = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                            'hours-per-week', 'native-country']

    QI = ['age', 'workclass', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
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
    encoded = pd.DataFrame(encoded)
    base_est = DecisionTreeClassifier(random_state=0, min_samples_split=2,
                                      min_samples_leaf=1)
    model = SklearnClassifier(base_est, ModelOutputType.CLASSIFIER_PROBABILITIES)
    model.fit(ArrayDataset(encoded, y_train))
    predictions = model.predict(ArrayDataset(encoded))
    if predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)
    target_accuracy = 0.7
    gen = GeneralizeToRepresentative(model, target_accuracy=target_accuracy,
                                     categorical_features=categorical_features, features_to_minimize=QI)
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

    for key in expected_generalizations['ranges']:
        assert (set(expected_generalizations['ranges'][key]) == set(gener['ranges'][key]))
    for key in expected_generalizations['categories']:
        assert (set([frozenset(sl) for sl in expected_generalizations['categories'][key]])
                == set([frozenset(sl) for sl in gener['categories'][key]]))
    assert (set(expected_generalizations['untouched']) == set(gener['untouched']))
    # assert (transformed.drop(QI, axis=1).equals(x_train.drop(QI, axis=1)))
    np.testing.assert_array_equal(transformed.drop(QI, axis=1), x_train.drop(QI, axis=1))

    modified_features = [f for f in features if
                         f in expected_generalizations['categories'].keys() or f in expected_generalizations[
                             'ranges'].keys()]
    # assert (transformed.drop(modified_features, axis=1).equals(x_train.drop(modified_features, axis=1)))
    np.testing.assert_array_equal(transformed.drop(modified_features, axis=1), x_train.drop(modified_features, axis=1))
    ncp = gen.ncp
    if len(expected_generalizations['ranges'].keys()) > 0 or len(expected_generalizations['categories'].keys()) > 0:
        assert (ncp > 0)
        assert (((transformed[modified_features]).equals(x_train[modified_features])) is False)

    rel_accuracy = model.score(ArrayDataset(preprocessor.transform(transformed), predictions))
    assert ((rel_accuracy >= target_accuracy) or (target_accuracy - rel_accuracy) <= 0.05)


def test_german_credit_pandas():
    (x_train, y_train), (x_test, y_test) = get_german_credit_dataset_pd()
    features = ["Existing_checking_account", "Duration_in_month", "Credit_history", "Purpose", "Credit_amount",
                "Savings_account", "Present_employment_since", "Installment_rate", "Personal_status_sex", "debtors",
                "Present_residence", "Property", "Age", "Other_installment_plans", "Housing",
                "Number_of_existing_credits", "Job", "N_people_being_liable_provide_maintenance", "Telephone",
                "Foreign_worker"]
    categorical_features = ["Existing_checking_account", "Credit_history", "Purpose", "Savings_account",
                            "Present_employment_since", "Personal_status_sex", "debtors", "Property",
                            "Other_installment_plans", "Housing", "Job"]
    QI = ["Duration_in_month", "Credit_history", "Purpose", "debtors", "Property", "Other_installment_plans",
          "Housing", "Job"]

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
    encoded = pd.DataFrame(encoded)
    base_est = DecisionTreeClassifier(random_state=0, min_samples_split=2,
                                      min_samples_leaf=1)
    model = SklearnClassifier(base_est, ModelOutputType.CLASSIFIER_PROBABILITIES)
    model.fit(ArrayDataset(encoded, y_train))
    predictions = model.predict(ArrayDataset(encoded))
    if predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)
    target_accuracy = 0.7
    gen = GeneralizeToRepresentative(model, target_accuracy=target_accuracy,
                                     categorical_features=categorical_features, features_to_minimize=QI)
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

    for key in expected_generalizations['ranges']:
        assert (set(expected_generalizations['ranges'][key]) == set(gener['ranges'][key]))
    for key in expected_generalizations['categories']:
        assert (set([frozenset(sl) for sl in expected_generalizations['categories'][key]])
                == set([frozenset(sl) for sl in gener['categories'][key]]))
    assert (set(expected_generalizations['untouched']) == set(gener['untouched']))
    # assert (transformed.drop(QI, axis=1).equals(x_train.drop(QI, axis=1)))
    np.testing.assert_array_equal(transformed.drop(QI, axis=1), x_train.drop(QI, axis=1))

    modified_features = [f for f in features if
                         f in expected_generalizations['categories'].keys() or f in expected_generalizations[
                             'ranges'].keys()]
    # assert (transformed.drop(modified_features, axis=1).equals(x_train.drop(modified_features, axis=1)))
    np.testing.assert_array_equal(transformed.drop(modified_features, axis=1), x_train.drop(modified_features, axis=1))
    ncp = gen.ncp
    if len(expected_generalizations['ranges'].keys()) > 0 or len(expected_generalizations['categories'].keys()) > 0:
        assert (ncp > 0)
        assert (((transformed[modified_features]).equals(x_train[modified_features])) is False)

    rel_accuracy = model.score(ArrayDataset(preprocessor.transform(transformed), predictions))
    assert ((rel_accuracy >= target_accuracy) or (target_accuracy - rel_accuracy) <= 0.05)


def test_regression():
    dataset = load_diabetes()
    x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.5, random_state=14)

    base_est = DecisionTreeRegressor(random_state=10, min_samples_split=2)
    model = SklearnRegressor(base_est)
    model.fit(ArrayDataset(x_train, y_train))
    predictions = model.predict(ArrayDataset(x_train))
    QI = ['age', 'bmi', 's2', 's5']
    features = ['age', 'sex', 'bmi', 'bp',
                's1', 's2', 's3', 's4', 's5', 's6']

    target_accuracy = 0.7
    gen = GeneralizeToRepresentative(model, target_accuracy=target_accuracy, is_regression=True,
                                     features_to_minimize=QI)
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

    for key in expected_generalizations['ranges']:
        assert (set(expected_generalizations['ranges'][key]) == set(gener['ranges'][key]))
    for key in expected_generalizations['categories']:
        assert (set([frozenset(sl) for sl in expected_generalizations['categories'][key]])
                == set([frozenset(sl) for sl in gener['categories'][key]]))
    assert (set(expected_generalizations['untouched']) == set(gener['untouched']))
    assert ((np.delete(transformed, [0, 2, 5, 8], axis=1) == np.delete(x_train, [0, 2, 5, 8], axis=1)).all())

    modified_features = [f for f in features if
                         f in expected_generalizations['categories'].keys() or f in expected_generalizations[
                             'ranges'].keys()]
    indexes = []
    for i in range(len(features)):
        if features[i] in modified_features:
            indexes.append(i)
    assert ((np.delete(transformed, indexes, axis=1) == np.delete(x_train, indexes, axis=1)).all())
    ncp = gen.ncp
    if len(expected_generalizations['ranges'].keys()) > 0 or len(expected_generalizations['categories'].keys()) > 0:
        assert (ncp > 0)
        assert (((transformed[indexes]) != (x_train[indexes])).any())

    rel_accuracy = model.score(ArrayDataset(transformed, predictions))
    assert ((rel_accuracy >= target_accuracy) or (target_accuracy - rel_accuracy) <= 0.05)


def test_X_y(data):
    features = [0, 1, 2]
    X = np.array([[23, 165, 70],
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
    print(X)
    y = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
    QI = [0, 2]
    base_est = DecisionTreeClassifier(random_state=0, min_samples_split=2,
                                      min_samples_leaf=1)
    model = SklearnClassifier(base_est, ModelOutputType.CLASSIFIER_PROBABILITIES)
    model.fit(ArrayDataset(X, y))
    ad = ArrayDataset(X)
    predictions = model.predict(ad)
    if predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)
    target_accuracy = 0.5
    gen = GeneralizeToRepresentative(model, target_accuracy=target_accuracy, features_to_minimize=QI)
    gen.fit(X=X, y=predictions)
    transformed = gen.transform(X)
    gener = gen.generalizations
    expected_generalizations = {'ranges': {'0': [], '2': [67.5]}, 'categories': {}, 'untouched': ['1']}
    for key in expected_generalizations['ranges']:
        assert (set(expected_generalizations['ranges'][key]) == set(gener['ranges'][key]))
    for key in expected_generalizations['categories']:
        assert (set([frozenset(sl) for sl in expected_generalizations['categories'][key]])
                == set([frozenset(sl) for sl in gener['categories'][key]]))
    assert (set(expected_generalizations['untouched']) == set(gener['untouched']))
    assert ((np.delete(transformed, [0, 2], axis=1) == np.delete(X, [0, 2], axis=1)).all())
    modified_features = [f for f in features if
                         str(f) in expected_generalizations['categories'].keys() or str(f) in expected_generalizations[
                             'ranges'].keys()]
    indexes = []
    for i in range(len(features)):
        if features[i] in modified_features:
            indexes.append(i)
    assert ((np.delete(transformed, indexes, axis=1) == np.delete(X, indexes, axis=1)).all())
    ncp = gen.ncp
    if len(expected_generalizations['ranges'].keys()) > 0 or len(expected_generalizations['categories'].keys()) > 0:
        assert (ncp > 0)
        assert (((transformed[indexes]) != (X[indexes])).any())

    rel_accuracy = model.score(ArrayDataset(transformed, predictions))
    assert ((rel_accuracy >= target_accuracy) or (target_accuracy - rel_accuracy) <= 0.05)


def test_X_y_features_names(data):
    features = ['age', 'height', 'weight']
    X = np.array([[23, 165, 70],
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
    print(X)
    y = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
    QI = ['age', 'weight']
    base_est = DecisionTreeClassifier(random_state=0, min_samples_split=2,
                                      min_samples_leaf=1)
    model = SklearnClassifier(base_est, ModelOutputType.CLASSIFIER_PROBABILITIES)
    model.fit(ArrayDataset(X, y))
    ad = ArrayDataset(X)
    predictions = model.predict(ad)
    if predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)
    target_accuracy = 0.5
    gen = GeneralizeToRepresentative(model, target_accuracy=target_accuracy, features_to_minimize=QI)
    gen.fit(X=X, y=predictions, features_names=features)
    transformed = gen.transform(X=X, features_names=features)
    gener = gen.generalizations
    expected_generalizations = {'ranges': {'age': [], 'weight': [67.5]}, 'categories': {}, 'untouched': ['height']}
    for key in expected_generalizations['ranges']:
        assert (set(expected_generalizations['ranges'][key]) == set(gener['ranges'][key]))
    for key in expected_generalizations['categories']:
        assert (set([frozenset(sl) for sl in expected_generalizations['categories'][key]])
                == set([frozenset(sl) for sl in gener['categories'][key]]))
    assert (set(expected_generalizations['untouched']) == set(gener['untouched']))
    assert ((np.delete(transformed, [0, 2], axis=1) == np.delete(X, [0, 2], axis=1)).all())
    modified_features = [f for f in features if
                         f in expected_generalizations['categories'].keys() or f in expected_generalizations[
                             'ranges'].keys()]
    indexes = []
    for i in range(len(features)):
        if features[i] in modified_features:
            indexes.append(i)
    assert ((np.delete(transformed, indexes, axis=1) == np.delete(X, indexes, axis=1)).all())
    ncp = gen.ncp
    if len(expected_generalizations['ranges'].keys()) > 0 or len(expected_generalizations['categories'].keys()) > 0:
        assert (ncp > 0)
        assert (((transformed[indexes]) != (X[indexes])).any())

    rel_accuracy = model.score(ArrayDataset(transformed, predictions))
    assert ((rel_accuracy >= target_accuracy) or (target_accuracy - rel_accuracy) <= 0.05)


def test_BaseEstimator_classification(data):
    features = ['age', 'height', 'weight', 'sex', 'ola']
    X = [[23, 165, 65, 'f', 'aa'],
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
    X = pd.DataFrame(X, columns=features)
    QI = ['age', 'weight', 'ola']

    numeric_features = ["age", "height", "weight"]
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
    encoded = pd.DataFrame(encoded)
    base_est = DecisionTreeClassifier(random_state=0, min_samples_split=2,
                                      min_samples_leaf=1)
    model = base_est
    model.fit(encoded, y)
    predictions = model.predict(encoded)

    # Append classifier to preprocessing pipeline.
    # Now we have a full prediction pipeline.
    target_accuracy = 0.5
    gen = GeneralizeToRepresentative(model, target_accuracy=target_accuracy,
                                     categorical_features=categorical_features, features_to_minimize=QI)
    train_dataset = ArrayDataset(X, predictions)
    gen.fit(dataset=train_dataset)
    transformed = gen.transform(dataset=ArrayDataset(X))
    gener = gen.generalizations
    expected_generalizations = {'ranges': {'age': [], 'weight': [47.0]}, 'categories': {'ola': [['bb', 'aa']]},
                                'untouched': ['height', 'sex']}

    for key in expected_generalizations['ranges']:
        assert (set(expected_generalizations['ranges'][key]) == set(gener['ranges'][key]))
    for key in expected_generalizations['categories']:
        assert (set([frozenset(sl) for sl in expected_generalizations['categories'][key]])
                == set([frozenset(sl) for sl in gener['categories'][key]]))
    assert (set(expected_generalizations['untouched']) == set(gener['untouched']))
    # assert (transformed.drop(QI, axis=1).equals(X.drop(QI, axis=1)))
    np.testing.assert_array_equal(transformed.drop(QI, axis=1), X.drop(QI, axis=1))
    modified_features = [f for f in features if
                         f in expected_generalizations['categories'].keys() or f in expected_generalizations[
                             'ranges'].keys()]
    # assert (transformed.drop(modified_features, axis=1).equals(X.drop(modified_features, axis=1)))
    np.testing.assert_array_equal(transformed.drop(modified_features, axis=1), X.drop(modified_features, axis=1))
    ncp = gen.ncp
    if len(expected_generalizations['ranges'].keys()) > 0 or len(expected_generalizations['categories'].keys()) > 0:
        assert (ncp > 0)
        assert (((transformed[modified_features]).equals(X[modified_features])) is False)

    rel_accuracy = model.score(preprocessor.transform(transformed), predictions)
    assert ((rel_accuracy >= target_accuracy) or (target_accuracy - rel_accuracy) <= 0.05)


def test_BaseEstimator_regression():
    dataset = load_diabetes()
    x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.5, random_state=14)

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

    for key in expected_generalizations['ranges']:
        assert (set(expected_generalizations['ranges'][key]) == set(gener['ranges'][key]))
    for key in expected_generalizations['categories']:
        assert (set([frozenset(sl) for sl in expected_generalizations['categories'][key]])
                == set([frozenset(sl) for sl in gener['categories'][key]]))
    assert (set(expected_generalizations['untouched']) == set(gener['untouched']))
    assert ((np.delete(transformed, [0, 2, 5, 8], axis=1) == np.delete(x_train, [0, 2, 5, 8], axis=1)).all())

    modified_features = [f for f in features if
                         f in expected_generalizations['categories'].keys() or f in expected_generalizations[
                             'ranges'].keys()]
    indexes = []
    for i in range(len(features)):
        if features[i] in modified_features:
            indexes.append(i)
    assert ((np.delete(transformed, indexes, axis=1) == np.delete(x_train, indexes, axis=1)).all())
    ncp = gen.ncp
    if len(expected_generalizations['ranges'].keys()) > 0 or len(expected_generalizations['categories'].keys()) > 0:
        assert (ncp > 0)
        assert (((transformed[indexes]) != (x_train[indexes])).any())

    rel_accuracy = model.score(transformed, predictions)
    assert ((rel_accuracy >= target_accuracy) or (target_accuracy - rel_accuracy) <= 0.05)


def test_keras_model():
    (X, y), (x_test, y_test) = get_iris_dataset_np()

    base_est = Sequential()
    base_est.add(Input(shape=(4,)))
    base_est.add(Dense(10, activation="relu"))
    base_est.add(Dense(3, activation='softmax'))

    base_est.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model = KerasClassifier(base_est, ModelOutputType.CLASSIFIER_PROBABILITIES)
    model.fit(ArrayDataset(X, y))
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
    modified_features = [f for f in features if
                         f in gener['categories'].keys() or f in gener['ranges'].keys()]
    indexes = []
    for i in range(len(features)):
        if features[i] in modified_features:
            indexes.append(i)
    assert ((np.delete(transformed, indexes, axis=1) == np.delete(x_test, indexes, axis=1)).all())
    ncp = gen.ncp
    if len(gener['ranges'].keys()) > 0 or len(gener['categories'].keys()) > 0:
        assert (ncp > 0)
        assert (((transformed[indexes]) != (X[indexes])).any())

    rel_accuracy = model.score(ArrayDataset(transformed, predictions))
    assert ((rel_accuracy >= target_accuracy) or (target_accuracy - rel_accuracy) <= 0.05)


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
    for key in expected_generalizations['ranges']:
        assert (set(expected_generalizations['ranges'][key]) == set(gener['ranges'][key]))
    for key in expected_generalizations['categories']:
        assert (set([frozenset(sl) for sl in expected_generalizations['categories'][key]])
                == set([frozenset(sl) for sl in gener['categories'][key]]))
    assert (set(expected_generalizations['untouched']) == set(gener['untouched']))
