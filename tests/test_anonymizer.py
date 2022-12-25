import pytest
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder

from apt.anonymization import Anonymize
from apt.utils.dataset_utils import get_iris_dataset_np, get_adult_dataset_pd, get_nursery_dataset_pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from apt.utils.datasets import ArrayDataset


def test_anonymize_ndarray_iris():
    (x_train, y_train), _ = get_iris_dataset_np()

    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    pred = model.predict(x_train)

    k = 10
    QI = [0, 2]
    anonymizer = Anonymize(k, QI, train_only_QI=True)
    anon = anonymizer.anonymize(ArrayDataset(x_train, pred))
    assert (len(np.unique(anon[:, QI], axis=0)) < len(np.unique(x_train[:, QI], axis=0)))
    _, counts_elements = np.unique(anon[:, QI], return_counts=True)
    assert (np.min(counts_elements) >= k)
    assert ((np.delete(anon, QI, axis=1) == np.delete(x_train, QI, axis=1)).all())


def test_anonymize_pandas_adult():
    (x_train, y_train), _ = get_adult_dataset_pd()

    k = 100
    features = ['age', 'workclass', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
    QI = ['age', 'workclass', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
          'native-country']
    categorical_features = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                            'native-country']
    # prepare data for DT
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
    model = DecisionTreeClassifier()
    model.fit(encoded, y_train)
    pred = model.predict(encoded)

    anonymizer = Anonymize(k, QI, categorical_features=categorical_features)
    anon = anonymizer.anonymize(ArrayDataset(x_train, pred, features))

    assert (anon.loc[:, QI].drop_duplicates().shape[0] < x_train.loc[:, QI].drop_duplicates().shape[0])
    assert (anon.loc[:, QI].value_counts().min() >= k)
    np.testing.assert_array_equal(anon.drop(QI, axis=1), x_train.drop(QI, axis=1))


def test_anonymize_pandas_nursery():
    (x_train, y_train), _ = get_nursery_dataset_pd()
    x_train = x_train.astype(str)

    k = 100
    features = ["parents", "has_nurs", "form", "children", "housing", "finance", "social", "health"]
    QI = ["finance", "social", "health"]
    categorical_features = ["parents", "has_nurs", "form", "housing", "finance", "social", "health", 'children']
    # prepare data for DT
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
    model = DecisionTreeClassifier()
    model.fit(encoded, y_train)
    pred = model.predict(encoded)

    anonymizer = Anonymize(k, QI, categorical_features=categorical_features, train_only_QI=True)
    anon = anonymizer.anonymize(ArrayDataset(x_train, pred))

    assert (anon.loc[:, QI].drop_duplicates().shape[0] < x_train.loc[:, QI].drop_duplicates().shape[0])
    assert (anon.loc[:, QI].value_counts().min() >= k)
    np.testing.assert_array_equal(anon.drop(QI, axis=1), x_train.drop(QI, axis=1))


def test_regression():
    dataset = load_diabetes()
    x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.5, random_state=14)

    model = DecisionTreeRegressor(random_state=10, min_samples_split=2)
    model.fit(x_train, y_train)
    pred = model.predict(x_train)
    k = 10
    QI = [0, 2, 5, 8]
    anonymizer = Anonymize(k, QI, is_regression=True, train_only_QI=True)
    anon = anonymizer.anonymize(ArrayDataset(x_train, pred))
    print('Base model accuracy (R2 score): ', model.score(x_test, y_test))
    model.fit(anon, y_train)
    print('Base model accuracy (R2 score) after anonymization: ', model.score(x_test, y_test))
    assert (len(np.unique(anon[:, QI], axis=0)) < len(np.unique(x_train[:, QI], axis=0)))
    _, counts_elements = np.unique(anon[:, QI], return_counts=True)
    assert (np.min(counts_elements) >= k)
    assert ((np.delete(anon, QI, axis=1) == np.delete(x_train, QI, axis=1)).all())


def test_errors():
    with pytest.raises(ValueError):
        Anonymize(1, [0, 2])
    with pytest.raises(ValueError):
        Anonymize(2, [])
    with pytest.raises(ValueError):
        Anonymize(2, None)
    anonymizer = Anonymize(10, [0, 2])
    (x_train, y_train), (x_test, y_test) = get_iris_dataset_np()
    with pytest.raises(ValueError):
        anonymizer.anonymize(dataset=ArrayDataset(x_train, y_test))
    (x_train, y_train), _ = get_adult_dataset_pd()
    with pytest.raises(ValueError):
        anonymizer.anonymize(dataset=ArrayDataset(x_train, y_test))
