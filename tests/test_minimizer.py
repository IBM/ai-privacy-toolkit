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
from apt.utils import get_iris_dataset, get_adult_dataset, get_nursery_dataset, get_german_credit_dataset


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
    print(X.dtype)
    y = [1, 1, 0]
    base_est = DecisionTreeClassifier(random_state=0, min_samples_split=2,
                                      min_samples_leaf=1)
    base_est.fit(X, y)

    gen = GeneralizeToRepresentative(base_est, features=features, cells=cells)
    gen.fit()
    transformed = gen.transform(X)



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
    expexted_generalizations = {'ranges': {}, 'categories': {}, 'untouched': ['height', 'age']}
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
    expexted_generalizations = {'ranges': {'age': []}, 'categories': {}, 'untouched': ['ola', 'height', 'sex']}
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
    y = [1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0]
    QI = [0, 2]
    base_est = DecisionTreeClassifier(random_state=0, min_samples_split=2,
                                      min_samples_leaf=1)
    base_est.fit(X, y)
    predictions = base_est.predict(X)

    gen = GeneralizeToRepresentative(base_est, features=features, target_accuracy=0.5, features_to_minimize=QI)
    gen.fit(X, predictions)
    transformed = gen.transform(X)
    gener = gen.generalizations_
    expexted_generalizations = {'ranges': {'age': [], 'weight': [67.5]}, 'categories': {}, 'untouched': ['height']}
    for key in expexted_generalizations['ranges']:
        assert (set(expexted_generalizations['ranges'][key]) == set(gener['ranges'][key]))
    for key in expexted_generalizations['categories']:
        assert (set([frozenset(sl) for sl in expexted_generalizations['categories'][key]]) ==
                set([frozenset(sl) for sl in gener['categories'][key]]))
    assert (set(expexted_generalizations['untouched']) == set(gener['untouched']))
    assert ((np.delete(transformed, QI, axis=1) == np.delete(X, QI, axis=1)).all())
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

    y = [1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0]
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
    base_est = DecisionTreeClassifier(random_state=0, min_samples_split=2,
                                      min_samples_leaf=1)
    base_est.fit(encoded, y)
    predictions = base_est.predict(encoded)
    # Append classifier to preprocessing pipeline.
    # Now we have a full prediction pipeline.
    gen = GeneralizeToRepresentative(base_est, features=features, target_accuracy=0.5,
                                     categorical_features=categorical_features, features_to_minimize=QI)
    gen.fit(X, predictions)
    transformed = gen.transform(X)
    gener = gen.generalizations_
    expexted_generalizations = {'ranges': {'age': [], 'weight': [47.0]}, 'categories': {'ola': [['bb', 'aa']]}, 'untouched': ['height', 'sex']}
    for key in expexted_generalizations['ranges']:
        assert (set(expexted_generalizations['ranges'][key]) == set(gener['ranges'][key]))
    for key in expexted_generalizations['categories']:
        assert (set([frozenset(sl) for sl in expexted_generalizations['categories'][key]]) ==
                set([frozenset(sl) for sl in gener['categories'][key]]))
    assert (set(expexted_generalizations['untouched']) == set(gener['untouched']))
    assert (transformed.drop(QI, axis=1).equals(X.drop(QI, axis=1)))
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

    QI = [0, 2]
    model = DecisionTreeClassifier(random_state=0, min_samples_split=2,
                                      min_samples_leaf=1)
    model.fit(x_train, y_train)
    pred = model.predict(x_train)

    gen = GeneralizeToRepresentative(model, target_accuracy=0.3, features=features, features_to_minimize=QI)
    gen.fit(x_train, pred)
    transformed = gen.transform(x_train)
    gener = gen.generalizations_
    expexted_generalizations = {'ranges': {'sepal length (cm)': [], 'petal length (cm)': [2.449999988079071]}, 'categories': {}, 'untouched': ['petal width (cm)', 'sepal width (cm)']}
    for key in expexted_generalizations['ranges']:
        assert (set(expexted_generalizations['ranges'][key]) == set(gener['ranges'][key]))
    for key in expexted_generalizations['categories']:
        assert (set([frozenset(sl) for sl in expexted_generalizations['categories'][key]]) ==
                set([frozenset(sl) for sl in gener['categories'][key]]))
    assert (set(expexted_generalizations['untouched']) == set(gener['untouched']))
    assert ((np.delete(transformed, QI, axis=1) == np.delete(x_train, QI, axis=1)).all())
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


def test_minimize_pandas_adult():
    (x_train, y_train), _ = get_adult_dataset()
    x_train = x_train.head(1000)
    y_train = y_train.head(1000)

    features = ['age', 'workclass', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']

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
    base_est = DecisionTreeClassifier(random_state=0, min_samples_split=2,
                                      min_samples_leaf=1)
    base_est.fit(encoded, y_train)
    predictions = base_est.predict(encoded)

    gen = GeneralizeToRepresentative(base_est, target_accuracy=0.7, features=features,
                                     categorical_features=categorical_features, features_to_minimize=QI)
    gen.fit(x_train, predictions)
    transformed = gen.transform(x_train)
    gener = gen.generalizations_
    expexted_generalizations = {'ranges': {'age': [], 'education-num': []}, 'categories': {'workclass': [['Self-emp-not-inc', 'Private', 'Federal-gov', 'Self-emp-inc', '?', 'Local-gov', 'State-gov']], 'marital-status': [['Divorced', 'Married-AF-spouse', 'Married-spouse-absent', 'Widowed', 'Separated', 'Married-civ-spouse', 'Never-married']], 'occupation': [['Tech-support', 'Priv-house-serv', 'Machine-op-inspct', 'Other-service', 'Prof-specialty', 'Adm-clerical', 'Protective-serv', 'Handlers-cleaners', 'Transport-moving', 'Armed-Forces', '?', 'Sales', 'Farming-fishing', 'Exec-managerial', 'Craft-repair']], 'relationship': [['Not-in-family', 'Wife', 'Other-relative', 'Husband', 'Unmarried', 'Own-child']], 'race': [['Asian-Pac-Islander', 'White', 'Other', 'Black', 'Amer-Indian-Eskimo']], 'sex': [['Female', 'Male']], 'native-country': [['Euro_1', 'LatinAmerica', 'BritishCommonwealth', 'SouthAmerica', 'UnitedStates', 'China', 'Euro_2', 'SE_Asia', 'Other', 'Unknown']]}, 'untouched': ['capital-loss', 'hours-per-week', 'capital-gain']}
    for key in expexted_generalizations['ranges']:
        assert (set(expexted_generalizations['ranges'][key]) == set(gener['ranges'][key]))
    for key in expexted_generalizations['categories']:
        assert (set([frozenset(sl) for sl in expexted_generalizations['categories'][key]]) ==
                set([frozenset(sl) for sl in gener['categories'][key]]))
    assert (set(expexted_generalizations['untouched']) == set(gener['untouched']))
    assert (transformed.drop(QI, axis=1).equals(x_train.drop(QI, axis=1)))
    modified_features = [f for f in features if
                         f in expexted_generalizations['categories'].keys() or f in expexted_generalizations[
                             'ranges'].keys()]
    assert (transformed.drop(modified_features, axis=1).equals(x_train.drop(modified_features, axis=1)))
    ncp = gen.ncp_
    if len(expexted_generalizations['ranges'].keys()) > 0 or len(expexted_generalizations['categories'].keys()) > 0:
        assert (ncp > 0)
        assert (((transformed[modified_features]).equals(x_train[modified_features])) == False)


def test_german_credit_pandas():
    (x_train, y_train), (x_test, y_test) = get_german_credit_dataset()
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
    base_est = DecisionTreeClassifier(random_state=0, min_samples_split=2,
                                      min_samples_leaf=1)
    base_est.fit(encoded, y_train)
    predictions = base_est.predict(encoded)

    gen = GeneralizeToRepresentative(base_est, target_accuracy=0.7, features=features,
                                     categorical_features=categorical_features, features_to_minimize=QI)
    gen.fit(x_train, predictions)
    transformed = gen.transform(x_train)
    gener = gen.generalizations_
    expexted_generalizations = {'ranges': {'Duration_in_month': [31.5]}, 'categories': {'Credit_history': [['A30', 'A32', 'A31', 'A34', 'A33']], 'Purpose': [['A41', 'A46', 'A43', 'A40', 'A44', 'A410', 'A49', 'A45', 'A48', 'A42']], 'debtors': [['A101', 'A102', 'A103']], 'Property': [['A124', 'A121', 'A122', 'A123']], 'Other_installment_plans': [['A142', 'A141', 'A143']], 'Housing': [['A151', 'A152', 'A153']], 'Job': [['A172', 'A171', 'A174', 'A173']]}, 'untouched': ['Installment_rate', 'Present_residence', 'Personal_status_sex', 'Foreign_worker', 'Telephone', 'Savings_account', 'Number_of_existing_credits', 'N_people_being_liable_provide_maintenance', 'Age', 'Existing_checking_account', 'Credit_amount', 'Present_employment_since']}
    for key in expexted_generalizations['ranges']:
        assert (set(expexted_generalizations['ranges'][key]) == set(gener['ranges'][key]))
    for key in expexted_generalizations['categories']:
        assert (set([frozenset(sl) for sl in expexted_generalizations['categories'][key]]) ==
                set([frozenset(sl) for sl in gener['categories'][key]]))
    assert (set(expexted_generalizations['untouched']) == set(gener['untouched']))
    assert (transformed.drop(QI, axis=1).equals(x_train.drop(QI, axis=1)))
    modified_features = [f for f in features if
                         f in expexted_generalizations['categories'].keys() or f in expexted_generalizations[
                             'ranges'].keys()]
    assert (transformed.drop(modified_features, axis=1).equals(x_train.drop(modified_features, axis=1)))
    ncp = gen.ncp_
    if len(expexted_generalizations['ranges'].keys()) > 0 or len(expexted_generalizations['categories'].keys()) > 0:
        assert (ncp > 0)
        assert (((transformed[modified_features]).equals(x_train[modified_features])) == False)

