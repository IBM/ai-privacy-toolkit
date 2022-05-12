import pytest

from apt.utils.models import SklearnClassifier, SklearnRegressor, ModelOutputType, KerasClassifier, BlackboxClassifier
from apt.utils.datasets import ArrayDataset, Data
from apt.utils import dataset_utils

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input


def test_sklearn_classifier():
    (x_train, y_train), (x_test, y_test) = dataset_utils.get_iris_dataset_np()
    underlying_model = RandomForestClassifier()
    model = SklearnClassifier(underlying_model, ModelOutputType.CLASSIFIER_PROBABILITIES)
    train = ArrayDataset(x_train, y_train)
    test = ArrayDataset(x_test, y_test)
    model.fit(train)
    pred = model.predict(test)
    assert(pred.shape[0] == x_test.shape[0])

    score = model.score(test)
    assert(0.0 <= score <= 1.0)


def test_sklearn_regressor():
    (x_train, y_train), (x_test, y_test) = dataset_utils.get_diabetes_dataset_np()
    underlying_model = DecisionTreeRegressor()
    model = SklearnRegressor(underlying_model)
    train = ArrayDataset(x_train, y_train)
    test = ArrayDataset(x_test, y_test)
    model.fit(train)
    pred = model.predict(test)
    assert (pred.shape[0] == x_test.shape[0])

    score = model.score(test)


def test_keras_classifier():
    (x_train, y_train), (x_test, y_test) = dataset_utils.get_iris_dataset_np()

    underlying_model = Sequential()
    underlying_model.add(Input(shape=(4,)))
    underlying_model.add(Dense(100, activation="relu"))
    underlying_model.add(Dense(10, activation="relu"))
    underlying_model.add(Dense(3, activation='softmax'))

    underlying_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model = KerasClassifier(underlying_model, ModelOutputType.CLASSIFIER_PROBABILITIES)

    train = ArrayDataset(x_train, y_train)
    test = ArrayDataset(x_test, y_test)
    model.fit(train)
    pred = model.predict(test)
    assert(pred.shape[0] == x_test.shape[0])

    score = model.score(test)
    assert(0.0 <= score <= 1.0)


def test_blackbox_classifier():
    (x_train, y_train), (x_test, y_test) = dataset_utils.get_iris_dataset_np()

    train = ArrayDataset(x_train, y_train)
    test = ArrayDataset(x_test, y_test)
    data = Data(train, test)
    model = BlackboxClassifier(data, ModelOutputType.CLASSIFIER_PROBABILITIES)
    pred = model.predict(test)
    assert(pred.shape[0] == x_test.shape[0])

    score = model.score(test)
    assert(0.0 <= score <= 1.0)
