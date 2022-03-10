import pytest

from apt.utils.models import SklearnClassifier, SklearnRegressor
from apt.utils.datasets import ArrayDataset
from apt.utils import dataset_utils

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier


def test_sklearn_classifier():
    dataset = dataset_utils.get_iris_dataset()
    underlying_model = RandomForestClassifier()
    model = SklearnClassifier(underlying_model)
    model.fit(dataset.train)
    pred = model.predict(dataset.get_test_samples())
    assert(pred.shape[0] == dataset.get_test_samples().shape[0])

    score = model.score(dataset.test)
    assert(0.0 <= score <= 1.0)


def test_sklearn_regressor():
    dataset = dataset_utils.get_diabetes_dataset()
    underlying_model = DecisionTreeRegressor()
    model = SklearnRegressor(underlying_model)
    model.fit(dataset.train)
    pred = model.predict(dataset.get_test_samples())
    assert (pred.shape[0] == dataset.get_test_samples().shape[0])

    score = model.score(dataset.test)
    assert (0 <= score <= 1)
