import pytest
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder

from apt.anonymization import Anonymize
from apt.utils.dataset_utils import get_iris_dataset, get_adult_dataset, get_nursery_dataset
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from apt.utils.datasets import ArrayDataset, Data

def test_anonymize_ndarray_iris():
    dataset = get_iris_dataset()
    model = DecisionTreeClassifier()
    model.fit(dataset.get_train_samples(), dataset.get_train_labels())
    pred = model.predict(dataset.get_train_samples())

    k = 10
    QI = [0, 2]
    anonymizer = Anonymize(k, QI)
    anon = anonymizer.anonymize(ArrayDataset(dataset.get_train_samples(), pred))
    assert(len(np.unique(anon[:, QI], axis=0)) < len(np.unique(dataset.get_train_samples()[:, QI], axis=0)))
    _, counts_elements = np.unique(anon[:, QI], return_counts=True)
    assert (np.min(counts_elements) >= k)
    assert ((np.delete(anon, QI, axis=1) == np.delete(dataset.get_train_samples(), QI, axis=1)).all())


def test_anonymize_pandas_adult():
    dataset = get_adult_dataset()
    encoded = OneHotEncoder().fit_transform(dataset.get_train_samples())
    model = DecisionTreeClassifier()
    model.fit(encoded, dataset.get_train_labels())
    pred = model.predict(encoded)

    k = 100
    QI = ['age', 'workclass', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
          'native-country']
    categorical_features = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                            'native-country']
    anonymizer = Anonymize(k, QI, categorical_features=categorical_features)
    anon = anonymizer.anonymize(ArrayDataset(dataset.get_train_samples(), pred))

    assert(anon.loc[:, QI].drop_duplicates().shape[0] < dataset.get_train_samples().loc[:, QI].drop_duplicates().shape[0])
    assert (anon.loc[:, QI].value_counts().min() >= k)
    assert (anon.drop(QI, axis=1).equals(dataset.get_train_samples().drop(QI, axis=1)))


def test_anonymize_pandas_nursery():
    dataset = get_nursery_dataset()
    encoded = OneHotEncoder().fit_transform(dataset.get_train_samples())
    model = DecisionTreeClassifier()
    model.fit(encoded, dataset.get_train_labels())
    pred = model.predict(encoded)

    k = 100
    QI = ["finance", "social", "health"]
    categorical_features = ["parents", "has_nurs", "form", "housing", "finance", "social", "health", 'children']
    anonymizer = Anonymize(k, QI, categorical_features=categorical_features)
    anon = anonymizer.anonymize(ArrayDataset(dataset.get_train_samples(), pred))

    assert(anon.loc[:, QI].drop_duplicates().shape[0] < dataset.get_train_samples().loc[:, QI].drop_duplicates().shape[0])
    assert (anon.loc[:, QI].value_counts().min() >= k)
    assert (anon.drop(QI, axis=1).equals(dataset.get_train_samples().drop(QI, axis=1)))


def test_regression():

    x_train, x_test, y_train, y_test = train_test_split(load_diabetes().data, load_diabetes().target, test_size=0.5, random_state=14)
    train_dataset = ArrayDataset(x_train, y_train)
    test_dataset = ArrayDataset(x_test, y_test)
    dataset = Data(train_dataset, test_dataset)
    model = DecisionTreeRegressor(random_state=10, min_samples_split=2)
    model.fit(dataset.get_train_samples(), dataset.get_train_labels())
    pred = model.predict(dataset.get_train_samples())
    k = 10
    QI = [0, 2, 5, 8]
    anonymizer = Anonymize(k, QI, is_regression=True)
    anon = anonymizer.anonymize(ArrayDataset(dataset.get_train_samples(), pred))
    print('Base model accuracy (R2 score): ', model.score(dataset.get_test_samples(), dataset.get_test_labels()))
    model.fit(anon, dataset.get_train_labels())
    print('Base model accuracy (R2 score) after anonymization: ', model.score(dataset.get_test_samples(), dataset.get_test_labels()))
    assert(len(np.unique(anon[:, QI], axis=0)) < len(np.unique(dataset.get_train_samples()[:, QI], axis=0)))
    _, counts_elements = np.unique(anon[:, QI], return_counts=True)
    assert (np.min(counts_elements) >= k)
    assert ((np.delete(anon, QI, axis=1) == np.delete(dataset.get_train_samples(), QI, axis=1)).all())


def test_errors():
    with pytest.raises(ValueError):
        Anonymize(1, [0, 2])
    with pytest.raises(ValueError):
        Anonymize(2, [])
    with pytest.raises(ValueError):
        Anonymize(2, None)
    anonymizer = Anonymize(10, [0, 2])
    dataset = get_iris_dataset()
    with pytest.raises(ValueError):
        anonymizer.anonymize(ArrayDataset(dataset.get_train_samples(), dataset.get_test_labels()))
    dataset = get_adult_dataset()
    with pytest.raises(ValueError):
        anonymizer.anonymize(ArrayDataset(dataset.get_train_samples(), dataset.get_train_labels()))
