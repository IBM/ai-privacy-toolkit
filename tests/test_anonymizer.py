import pytest
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder

from apt.anonymization import Anonymize
from apt.utils import get_iris_dataset, get_adult_dataset, get_nursery_dataset
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# from art.estimators.regression.scikitlearn import ScikitlearnRegressor


def test_anonymize_ndarray_iris():
    (x_train, y_train), _ = get_iris_dataset()
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    pred = model.predict(x_train)

    k = 10
    QI = [0, 2]
    anonymizer = Anonymize(k, QI)
    anon = anonymizer.anonymize(x_train, pred)

    assert(len(np.unique(anon, axis=0)) < len(np.unique(x_train, axis=0)))
    _, counts_elements = np.unique(anon[:, QI], return_counts=True)
    assert (np.min(counts_elements) >= k)
    assert ((np.delete(anon, QI, axis=1) == np.delete(x_train, QI, axis=1)).all())


def test_anonymize_pandas_adult():
    (x_train, y_train), _ = get_adult_dataset()
    encoded = OneHotEncoder().fit_transform(x_train)
    model = DecisionTreeClassifier()
    model.fit(encoded, y_train)
    pred = model.predict(encoded)

    k = 100
    QI = ['age', 'workclass', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
          'native-country']
    categorical_features = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                            'native-country']
    anonymizer = Anonymize(k, QI, categorical_features=categorical_features)
    anon = anonymizer.anonymize(x_train, pred)

    assert(anon.drop_duplicates().shape[0] < x_train.drop_duplicates().shape[0])
    assert (anon.loc[:, QI].value_counts().min() >= k)
    assert (anon.drop(QI, axis=1).equals(x_train.drop(QI, axis=1)))


def test_anonymize_pandas_nursery():
    (x_train, y_train), _ = get_nursery_dataset()
    x_train = x_train.astype(str)
    encoded = OneHotEncoder().fit_transform(x_train)
    model = DecisionTreeClassifier()
    model.fit(encoded, y_train)
    pred = model.predict(encoded)

    k = 100
    QI = ["finance", "social", "health"]
    categorical_features = ["parents", "has_nurs", "form", "housing", "finance", "social", "health", 'children']
    anonymizer = Anonymize(k, QI, categorical_features=categorical_features)
    anon = anonymizer.anonymize(x_train, pred)

    assert(anon.drop_duplicates().shape[0] < x_train.drop_duplicates().shape[0])
    assert (anon.loc[:, QI].value_counts().min() >= k)
    assert (anon.drop(QI, axis=1).equals(x_train.drop(QI, axis=1)))


def test_regression():

    dataset = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.5, random_state=14)

    model = LinearRegression()
    model.fit(X_train, y_train)

    print('Base model accuracy (R2 score): ', model.score(X_test, y_test))


def test_errors():
    with pytest.raises(ValueError):
        Anonymize(1, [0, 2])
    with pytest.raises(ValueError):
        Anonymize(2, [])
    with pytest.raises(ValueError):
        Anonymize(2, None)
    anonymizer = Anonymize(10, [0, 2])
    (x_train, y_train), (x_test, y_test) = get_iris_dataset()
    with pytest.raises(ValueError):
        anonymizer.anonymize(x_train, y_test)
    (x_train, y_train), _ = get_adult_dataset()
    with pytest.raises(ValueError):
        anonymizer.anonymize(x_train, y_train)
