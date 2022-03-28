import pytest

from apt.utils.models import SklearnClassifier, SklearnRegressor, ModelOutputType
from apt.utils.datasets import ArrayDataset
from apt.utils import dataset_utils

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier


def test_sklearn_classifier():
    (x_train, y_train), (x_test, y_test) = dataset_utils.get_iris_dataset()
    underlying_model = RandomForestClassifier()
    model = SklearnClassifier(underlying_model, ModelOutputType.CLASSIFIER_VECTOR)
    train = ArrayDataset(x_train, y_train)
    test = ArrayDataset(x_test, y_test)
    model.fit(train)
    pred = model.predict(x_test)
    assert(pred.shape[0] == x_test.shape[0])

    score = model.score(test)
    assert(0.0 <= score <= 1.0)


def test_sklearn_regressor():
    (x_train, y_train), (x_test, y_test) = dataset_utils.get_diabetes_dataset()
    underlying_model = DecisionTreeRegressor()
    model = SklearnRegressor(underlying_model)
    train = ArrayDataset(x_train, y_train)
    test = ArrayDataset(x_test, y_test)
    model.fit(train)
    pred = model.predict(x_test)
    assert (pred.shape[0] == x_test.shape[0])

    score = model.score(test)
    assert (0 <= score <= 1)
