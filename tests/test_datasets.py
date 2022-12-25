import numpy as np

from apt.utils.datasets import Data, DatasetWithPredictions
from apt.utils import dataset_utils


def test_dataset_predictions():
    (x_train, y_train), (_, _) = dataset_utils.get_iris_dataset_np()
    pred = np.array([[0.23, 0.56, 0.21] for i in range(105)])

    dataset = DatasetWithPredictions(pred)
    data = Data(train=dataset)

    new_pred = data.get_train_set().get_predictions()

    assert np.equal(pred, new_pred).all()


def test_dataset_predictions_x():
    (x_train, y_train), (_, _) = dataset_utils.get_iris_dataset_np()
    pred = np.array([[0.23, 0.56, 0.21] for i in range(105)])

    dataset = DatasetWithPredictions(pred, x=x_train)
    data = Data(train=dataset)

    new_pred = data.get_train_set().get_predictions()

    assert np.equal(pred, new_pred).all()


def test_dataset_predictions_y():
    (x_train, y_train), (_, _) = dataset_utils.get_iris_dataset_np()
    pred = np.array([[0.23, 0.56, 0.21] for i in range(105)])

    dataset = DatasetWithPredictions(pred, x=x_train, y=y_train)
    data = Data(train=dataset)

    new_pred = data.get_train_set().get_predictions()

    assert np.equal(pred, new_pred).all()
