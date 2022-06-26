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

def test_blackbox_classifier_no_test():
    (x_train, y_train), (_, _) = dataset_utils.get_iris_dataset_np()

    train = ArrayDataset(x_train, y_train)

    data = Data(train)
    model = BlackboxClassifier(data, ModelOutputType.CLASSIFIER_PROBABILITIES)
    pred = model.predict(train)
    assert(pred.shape[0] == x_train.shape[0])

    score = model.score(train)
    assert(0.0 <= score <= 1.0)


def test_blackbox_classifier_no_train():
    (_, _), (x_test, y_test) = dataset_utils.get_iris_dataset_np()

    test = ArrayDataset(x_test, y_test)
    data = Data(test=test)
    model = BlackboxClassifier(data, ModelOutputType.CLASSIFIER_PROBABILITIES)
    pred = model.predict(test)
    assert(pred.shape[0] == x_test.shape[0])

    score = model.score(test)
    assert(0.0 <= score <= 1.0)


def test_blackbox_classifier_no_test_y():
    (x_train, y_train), (x_test, _) = dataset_utils.get_iris_dataset_np()

    train = ArrayDataset(x_train, y_train)
    test = ArrayDataset(x_test)
    data = Data(train, test)
    model = BlackboxClassifier(data, ModelOutputType.CLASSIFIER_PROBABILITIES)
    pred = model.predict(train)
    assert(pred.shape[0] == x_train.shape[0])

    score = model.score(train)
    assert(0.0 <= score <= 1.0)

    # since no test_y, BBC should use only test thus predict test should fail
    unable_to_predict_test = False
    try:
        model.predict(test)
    except  BaseException:
        unable_to_predict_test = True

    assert (unable_to_predict_test, True)

def test_blackbox_classifier_no_train_y():
    (x_train, _), (x_test, y_test) = dataset_utils.get_iris_dataset_np()

    train = ArrayDataset(x_train)
    test = ArrayDataset(x_test, y_test)
    data = Data(train, test)
    model = BlackboxClassifier(data, ModelOutputType.CLASSIFIER_PROBABILITIES)
    pred = model.predict(test)
    assert (pred.shape[0] == x_test.shape[0])

    score = model.score(test)
    assert (0.0 <= score <= 1.0)

    # since no train_y, BBC should use only test thus predict train should fail
    unable_to_predict_train = False
    try:
        model.predict(train)
    except BaseException:
        unable_to_predict_train = True

    assert(unable_to_predict_train,True)

