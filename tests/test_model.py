import pytest
import numpy as np

from apt.utils.models import SklearnClassifier, SklearnRegressor, ModelOutputType, KerasClassifier, KerasRegressor, \
    BlackboxClassifierPredictions, BlackboxClassifierPredictFunction, is_one_hot, get_nb_classes, XGBoostClassifier
from apt.utils.datasets import ArrayDataset, Data, DatasetWithPredictions
from apt.utils import dataset_utils

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

from art.utils import check_and_transform_label_format

from art.utils import to_categorical

tf.compat.v1.disable_eager_execution()


def test_sklearn_classifier():
    (x_train, y_train), (x_test, y_test) = dataset_utils.get_iris_dataset_np()
    underlying_model = RandomForestClassifier()
    model = SklearnClassifier(underlying_model, ModelOutputType.CLASSIFIER_PROBABILITIES)
    train = ArrayDataset(x_train, y_train)
    test = ArrayDataset(x_test, y_test)
    model.fit(train)
    pred = model.predict(test)
    assert (pred.shape[0] == x_test.shape[0])

    score = model.score(test)
    assert (0.0 <= score <= 1.0)


def test_sklearn_regressor():
    (x_train, y_train), (x_test, y_test) = dataset_utils.get_diabetes_dataset_np()
    underlying_model = DecisionTreeRegressor()
    model = SklearnRegressor(underlying_model)
    train = ArrayDataset(x_train, y_train)
    test = ArrayDataset(x_test, y_test)
    model.fit(train)
    pred = model.predict(test)
    assert (pred.shape[0] == x_test.shape[0])

    model.score(test)


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
    assert (pred.shape[0] == x_test.shape[0])

    score = model.score(test)
    assert (0.0 <= score <= 1.0)


def test_keras_regressor():
    (x_train, y_train), (x_test, y_test) = dataset_utils.get_diabetes_dataset_np()

    underlying_model = Sequential()
    underlying_model.add(Input(shape=(10,)))
    underlying_model.add(Dense(100, activation="relu"))
    underlying_model.add(Dense(10, activation="relu"))
    underlying_model.add(Dense(1))

    underlying_model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])

    model = KerasRegressor(underlying_model)

    train = ArrayDataset(x_train, y_train)
    test = ArrayDataset(x_test, y_test)
    model.fit(train)
    pred = model.predict(test)
    assert (pred.shape[0] == x_test.shape[0])

    model.score(test)


def test_xgboost_classifier():
    (x_train, y_train), (x_test, y_test) = dataset_utils.get_iris_dataset_np()
    underlying_model = XGBClassifier()
    underlying_model.fit(x_train, y_train)
    model = XGBoostClassifier(underlying_model, ModelOutputType.CLASSIFIER_PROBABILITIES, input_shape=(4,), nb_classes=3)
    train = ArrayDataset(x_train, y_train)
    test = ArrayDataset(x_test, y_test)
    pred = model.predict(test)
    assert (pred.shape[0] == x_test.shape[0])

    score = model.score(test)
    assert (0.0 <= score <= 1.0)

    model.fit(train)


def test_blackbox_classifier():
    (x_train, y_train), (x_test, y_test) = dataset_utils.get_iris_dataset_np()

    train = ArrayDataset(x_train, y_train)
    test = ArrayDataset(x_test, y_test)
    data = Data(train, test)
    model = BlackboxClassifierPredictions(data, ModelOutputType.CLASSIFIER_SCALAR)
    pred = model.predict(test)
    assert (pred.shape[0] == x_test.shape[0])

    score = model.score(test)
    assert (score == 1.0)

    assert model.model_type is None


def test_blackbox_classifier_predictions():
    (x_train, y_train), (x_test, y_test) = dataset_utils.get_iris_dataset_np()

    train = DatasetWithPredictions(y_train, x_train)
    test = DatasetWithPredictions(y_test, x_test)
    data = Data(train, test)
    model = BlackboxClassifierPredictions(data, ModelOutputType.CLASSIFIER_SCALAR)
    pred = model.predict(test)
    assert (pred.shape[0] == x_test.shape[0])
    assert model.model_type is None

    with pytest.raises(ValueError):
        model.score(test)


def test_blackbox_classifier_predictions_y():
    (x_train, y_train), (x_test, y_test) = dataset_utils.get_iris_dataset_np()

    train = DatasetWithPredictions(y_train, x_train, y_train)
    test = DatasetWithPredictions(y_test, x_test, y_test)
    data = Data(train, test)
    model = BlackboxClassifierPredictions(data, ModelOutputType.CLASSIFIER_SCALAR)
    pred = model.predict(test)
    assert (pred.shape[0] == x_test.shape[0])

    score = model.score(test)
    assert (score == 1.0)

    assert model.model_type is None


def test_blackbox_classifier_mismatch():
    (x_train, y_train), (x_test, y_test) = dataset_utils.get_iris_dataset_np()

    train = ArrayDataset(x_train, y_train)
    test = ArrayDataset(x_test, y_test)
    data = Data(train, test)
    with pytest.raises(ValueError):
        BlackboxClassifierPredictions(data, ModelOutputType.CLASSIFIER_PROBABILITIES)


def test_blackbox_classifier_no_test():
    (x_train, y_train), (_, _) = dataset_utils.get_iris_dataset_np()

    train = ArrayDataset(x_train, y_train)

    data = Data(train)
    model = BlackboxClassifierPredictions(data, ModelOutputType.CLASSIFIER_SCALAR)
    pred = model.predict(train)
    assert (pred.shape[0] == x_train.shape[0])

    score = model.score(train)
    assert (score == 1.0)

    predictions_x, predictions_y = model.get_predictions()
    assert np.array_equal(predictions_x, x_train)
    assert np.array_equal(predictions_y, check_and_transform_label_format(y_train, nb_classes=3))


def test_blackbox_classifier_no_train():
    (_, _), (x_test, y_test) = dataset_utils.get_iris_dataset_np()

    test = ArrayDataset(x_test, y_test)
    data = Data(test=test)
    model = BlackboxClassifierPredictions(data, ModelOutputType.CLASSIFIER_SCALAR)
    pred = model.predict(test)
    assert (pred.shape[0] == x_test.shape[0])

    score = model.score(test)
    assert (score == 1.0)

    predictions_x, predictions_y = model.get_predictions()
    assert np.array_equal(predictions_x, x_test)
    assert np.array_equal(predictions_y, check_and_transform_label_format(y_test, nb_classes=3))


def test_blackbox_classifier_no_test_y():
    (x_train, y_train), (x_test, _) = dataset_utils.get_iris_dataset_np()

    train = ArrayDataset(x_train, y_train)
    test = ArrayDataset(x_test)
    data = Data(train, test)
    model = BlackboxClassifierPredictions(data, ModelOutputType.CLASSIFIER_SCALAR)
    pred = model.predict(train)
    assert (pred.shape[0] == x_train.shape[0])

    score = model.score(train)
    assert (score == 1.0)

    # since no test_y, BBC should use only test thus predict test should fail
    unable_to_predict_test = False
    try:
        model.predict(test)
    except BaseException:
        unable_to_predict_test = True

    assert unable_to_predict_test


def test_blackbox_classifier_no_train_y():
    (x_train, _), (x_test, y_test) = dataset_utils.get_iris_dataset_np()

    train = ArrayDataset(x_train)
    test = ArrayDataset(x_test, y_test)
    data = Data(train, test)
    model = BlackboxClassifierPredictions(data, ModelOutputType.CLASSIFIER_SCALAR)
    pred = model.predict(test)
    assert (pred.shape[0] == x_test.shape[0])

    score = model.score(test)
    assert (score == 1.0)

    # since no train_y, BBC should use only test thus predict train should fail
    unable_to_predict_train = False
    try:
        model.predict(train)
    except BaseException:
        unable_to_predict_train = True

    assert unable_to_predict_train


def test_blackbox_classifier_probabilities():
    (x_train, _), (_, _) = dataset_utils.get_iris_dataset_np()
    y_train = np.array([[0.23, 0.56, 0.21] for i in range(105)])

    train = ArrayDataset(x_train, y_train)

    data = Data(train)
    model = BlackboxClassifierPredictions(data, ModelOutputType.CLASSIFIER_PROBABILITIES)
    pred = model.predict(train)
    assert (pred.shape[0] == x_train.shape[0])
    assert (0.0 < pred).all()
    assert (pred < 1.0).all()

    score = model.score(train)
    assert (score == 1.0)


def test_blackbox_classifier_predict():
    def predict(x):
        return np.array([[0.23, 0.56, 0.21] for i in range(x.shape[0])])

    (x_train, y_train), (_, _) = dataset_utils.get_iris_dataset_np()
    y_train = np.array([[0.23, 0.56, 0.21] for i in range(105)])

    train = ArrayDataset(x_train, y_train)

    model = BlackboxClassifierPredictFunction(predict, ModelOutputType.CLASSIFIER_PROBABILITIES, (4,), 3)
    pred = model.predict(train)
    assert (pred.shape[0] == x_train.shape[0])
    assert (0.0 < pred).all()
    assert (pred < 1.0).all()

    score = model.score(train)
    assert (score == 1.0)


def test_blackbox_classifier_predict_scalar():
    def predict(x):
        return np.array([[1.0] for i in range(x.shape[0])])

    (x_train, y_train), (_, _) = dataset_utils.get_iris_dataset_np()
    y_train = np.array([[0, 1, 0] for i in range(105)])

    train = ArrayDataset(x_train, y_train)

    model = BlackboxClassifierPredictFunction(predict, ModelOutputType.CLASSIFIER_SCALAR, (4,), 3)
    pred = model.predict(train)
    assert (pred.shape[0] == x_train.shape[0])

    score = model.score(train)
    assert (score == 1.0)


def test_is_one_hot():
    (_, y_train), (_, _) = dataset_utils.get_iris_dataset_np()

    assert (not is_one_hot(y_train))
    assert (not is_one_hot(y_train.reshape(-1, 1)))
    assert (is_one_hot(to_categorical(y_train)))


def test_get_nb_classes():
    (_, y_train), (_, y_test) = dataset_utils.get_iris_dataset_np()

    # shape: (x,) - not 1-hot
    nb_classes_test = get_nb_classes(y_test)
    nb_classes_train = get_nb_classes(y_train)
    assert (nb_classes_test == nb_classes_train)
    assert (nb_classes_test == 3)

    # shape: (x,1) - not 1-hot
    nb_classes_test = get_nb_classes(y_test.reshape(-1, 1))
    assert (nb_classes_test == 3)

    # shape: (x,3) - 1-hot
    y = to_categorical(y_test)
    nb_classes = get_nb_classes(y)
    assert (nb_classes == 3)

    # gaps: 1,2,4 (0,3 missing)
    y_test[y_test == 0] = 4
    nb_classes = get_nb_classes(y_test)
    assert (nb_classes == 5)
