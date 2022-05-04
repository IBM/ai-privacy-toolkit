import pytest

from apt.utils.models import SklearnClassifier, SklearnRegressor, ModelOutputType, KerasClassifier
from apt.utils.datasets import ArrayDataset
from apt.utils import dataset_utils

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Softmax, Input


def test_sklearn_classifier():
    (x_train, y_train), (x_test, y_test) = dataset_utils.get_iris_dataset()
    underlying_model = RandomForestClassifier()
    model = SklearnClassifier(underlying_model, ModelOutputType.CLASSIFIER_VECTOR)
    train = ArrayDataset(x_train, y_train)
    test = ArrayDataset(x_test, y_test)
    model.fit(train)
    pred = model.predict(test)
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
    pred = model.predict(test)
    assert (pred.shape[0] == x_test.shape[0])

    score = model.score(test)


def test_keras_classifier():
    (x_train, y_train), (x_test, y_test) = dataset_utils.get_iris_dataset()

    underlying_model = Sequential()
    underlying_model.add(Input(shape=(4,)))
    underlying_model.add(Dense(100, activation="relu"))
    underlying_model.add(Dense(10, activation="relu"))
    underlying_model.add(Dense(3, activation='softmax'))

    underlying_model.compile(loss="categorical_crossentropy", optimizer="adam",
                             metrics=["accuracy"])
    # underlying_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.01),
    #                          metrics=["accuracy"])

    # sometimes required for wrapper to work
    from sklearn.preprocessing import OneHotEncoder

    encoder = OneHotEncoder(sparse=False)
    y_encoded = encoder.fit_transform(y_train.reshape(-1, 1))
    underlying_model.fit(x_train, y_encoded, epochs=10)

    # model = KerasClassifier(underlying_model, ModelOutputType.CLASSIFIER_VECTOR)
    #
    # train = ArrayDataset(x_train, y_train)
    # test = ArrayDataset(x_test, y_test)
    # model.fit(train)
    # pred = model.predict(test)
    # assert(pred.shape[0] == x_test.shape[0])
    #
    # score = model.score(test)
    # assert(0.0 <= score <= 1.0)
