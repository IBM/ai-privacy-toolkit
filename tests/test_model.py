import pytest

from apt.utils.models import SklearnClassifier, SklearnRegressor
from apt.utils import dataset_utils

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier

def test_sklearn_classifier():
    dataset = dataset_utils.get_iris_dataset()
    underlying_model = RandomForestClassifier()
    model = SklearnClassifier(underlying_model)
    model.fit(dataset.get_train_samples(), dataset.get_train_labels())
    pred = model.predict(dataset.get_test_samples())
    assert(pred.shape[0] == dataset.get_test_samples().shape[0])

    score = model.score(dataset.get_test_samples(), dataset.get_test_labels())
    assert(0.0 <= score <= 1.0)

def test_sklearn_regressor():
    dataset = dataset_utils.get_diabetes_dataset()
    underlying_model = DecisionTreeRegressor()
    model = SklearnRegressor(underlying_model)
    model.fit(dataset.get_train_samples(), dataset.get_train_labels())
    pred = model.predict(dataset.get_test_samples())
    assert (pred.shape[0] == dataset.get_test_samples().shape[0])

    score = model.score(dataset.get_test_samples(), dataset.get_test_labels())

    losses = model.loss(dataset.get_test_samples(), dataset.get_test_labels())
    assert (losses.shape[0] == dataset.get_test_samples().shape[0])


# Probably not needed for now, as we will not be using these wrappers directly in ART.
# def test_sklearn_decision_tree():
#     (x_train, y_train), (x_test, y_test) = dataset_utils.get_iris_dataset()
#     underlying_model = DecisionTreeClassifier()
#     model = SklearnDecisionTreeClassifier(underlying_model)
#     model.fit(x_train, y_train)
#     pred = model.predict(x_test)
#     assert(pred.shape[0] == x_test.shape[0])
#
#     score = model.score(x_test, y_test)
#     assert(0.0 <= score <= 1.0)
