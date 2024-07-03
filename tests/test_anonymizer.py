import pytest
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from torch import nn, optim, sigmoid, where
from torch.nn import functional
from scipy.special import expit

from apt.utils.datasets.datasets import PytorchData
from apt.utils.models import CLASSIFIER_MULTI_OUTPUT_BINARY_LOGITS
from apt.utils.models.pytorch_model import PyTorchClassifier
from apt.anonymization import Anonymize
from apt.utils.dataset_utils import get_iris_dataset_np, get_adult_dataset_pd, get_nursery_dataset_pd
from apt.utils.datasets import ArrayDataset


def test_anonymize_ndarray_iris():
    (x_train, y_train), _ = get_iris_dataset_np()

    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    pred = model.predict(x_train)

    k = 10
    QI = [0, 2]
    anonymizer = Anonymize(k, QI, train_only_QI=True)
    anon = anonymizer.anonymize(ArrayDataset(x_train, pred))
    assert (len(np.unique(anon[:, QI], axis=0)) < len(np.unique(x_train[:, QI], axis=0)))
    _, counts_elements = np.unique(anon[:, QI], return_counts=True)
    assert (np.min(counts_elements) >= k)
    assert ((np.delete(anon, QI, axis=1) == np.delete(x_train, QI, axis=1)).all())


def test_anonymize_pandas_adult():
    (x_train, y_train), _ = get_adult_dataset_pd()

    k = 100
    features = ['age', 'workclass', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
    QI = ['age', 'workclass', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
          'native-country']
    categorical_features = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                            'native-country']
    # prepare data for DT
    numeric_features = [f for f in features if f not in categorical_features]
    numeric_transformer = Pipeline(
        steps=[('imputer', SimpleImputer(strategy='constant', fill_value=0))]
    )
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    encoded = preprocessor.fit_transform(x_train)
    model = DecisionTreeClassifier()
    model.fit(encoded, y_train)
    pred = model.predict(encoded)

    anonymizer = Anonymize(k, QI, categorical_features=categorical_features)
    anon = anonymizer.anonymize(ArrayDataset(x_train, pred, features))

    assert (anon.loc[:, QI].drop_duplicates().shape[0] < x_train.loc[:, QI].drop_duplicates().shape[0])
    assert (anon.loc[:, QI].value_counts().min() >= k)
    np.testing.assert_array_equal(anon.drop(QI, axis=1), x_train.drop(QI, axis=1))


def test_anonymize_pandas_nursery():
    (x_train, y_train), _ = get_nursery_dataset_pd()
    x_train = x_train.astype(str)

    k = 100
    features = ["parents", "has_nurs", "form", "children", "housing", "finance", "social", "health"]
    QI = ["finance", "social", "health"]
    categorical_features = ["parents", "has_nurs", "form", "housing", "finance", "social", "health", 'children']
    # prepare data for DT
    numeric_features = [f for f in features if f not in categorical_features]
    numeric_transformer = Pipeline(
        steps=[('imputer', SimpleImputer(strategy='constant', fill_value=0))]
    )
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    encoded = preprocessor.fit_transform(x_train)
    model = DecisionTreeClassifier()
    model.fit(encoded, y_train)
    pred = model.predict(encoded)

    anonymizer = Anonymize(k, QI, categorical_features=categorical_features, train_only_QI=True)
    anon = anonymizer.anonymize(ArrayDataset(x_train, pred))

    assert (anon.loc[:, QI].drop_duplicates().shape[0] < x_train.loc[:, QI].drop_duplicates().shape[0])
    assert (anon.loc[:, QI].value_counts().min() >= k)
    np.testing.assert_array_equal(anon.drop(QI, axis=1), x_train.drop(QI, axis=1))


def test_regression():
    dataset = load_diabetes()
    x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.5, random_state=14)

    model = DecisionTreeRegressor(random_state=10, min_samples_split=2)
    model.fit(x_train, y_train)
    pred = model.predict(x_train)
    k = 10
    QI = [0, 2, 5, 8]
    anonymizer = Anonymize(k, QI, is_regression=True, train_only_QI=True)
    anon = anonymizer.anonymize(ArrayDataset(x_train, pred))
    print('Base model accuracy (R2 score): ', model.score(x_test, y_test))
    model.fit(anon, y_train)
    print('Base model accuracy (R2 score) after anonymization: ', model.score(x_test, y_test))
    assert (len(np.unique(anon[:, QI], axis=0)) < len(np.unique(x_train[:, QI], axis=0)))
    _, counts_elements = np.unique(anon[:, QI], return_counts=True)
    assert (np.min(counts_elements) >= k)
    assert ((np.delete(anon, QI, axis=1) == np.delete(x_train, QI, axis=1)).all())


def test_anonymize_ndarray_one_hot():
    x_train = np.array([[23, 0, 1, 165],
                        [45, 0, 1, 158],
                        [56, 1, 0, 123],
                        [67, 0, 1, 154],
                        [45, 1, 0, 149],
                        [42, 1, 0, 166],
                        [73, 0, 1, 172],
                        [94, 0, 1, 168],
                        [69, 0, 1, 175],
                        [24, 1, 0, 181],
                        [18, 1, 0, 190]])
    y_train = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0])

    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    pred = model.predict(x_train)

    k = 10
    QI = [0, 1, 2]
    QI_slices = [[1, 2]]
    anonymizer = Anonymize(k, QI, train_only_QI=True, quasi_identifer_slices=QI_slices)
    anon = anonymizer.anonymize(ArrayDataset(x_train, pred))
    assert (len(np.unique(anon[:, QI], axis=0)) < len(np.unique(x_train[:, QI], axis=0)))
    _, counts_elements = np.unique(anon[:, QI], return_counts=True)
    assert (np.min(counts_elements) >= k)
    assert ((np.delete(anon, QI, axis=1) == np.delete(x_train, QI, axis=1)).all())
    anonymized_slice = anon[:, QI_slices[0]]
    assert ((np.sum(anonymized_slice, axis=1) == 1).all())
    assert ((np.max(anonymized_slice, axis=1) == 1).all())
    assert ((np.min(anonymized_slice, axis=1) == 0).all())


def test_anonymize_pandas_one_hot():
    feature_names = ["age", "gender_M", "gender_F", "height"]
    x_train = np.array([[23, 0, 1, 165],
                        [45, 0, 1, 158],
                        [56, 1, 0, 123],
                        [67, 0, 1, 154],
                        [45, 1, 0, 149],
                        [42, 1, 0, 166],
                        [73, 0, 1, 172],
                        [94, 0, 1, 168],
                        [69, 0, 1, 175],
                        [24, 1, 0, 181],
                        [18, 1, 0, 190]])
    y_train = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
    x_train = pd.DataFrame(x_train, columns=feature_names)
    y_train = pd.Series(y_train)

    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    pred = model.predict(x_train)

    k = 10
    QI = ["age", "gender_M", "gender_F"]
    QI_slices = [["gender_M", "gender_F"]]
    anonymizer = Anonymize(k, QI, train_only_QI=True, quasi_identifer_slices=QI_slices)
    anon = anonymizer.anonymize(ArrayDataset(x_train, pred))
    assert (anon.loc[:, QI].drop_duplicates().shape[0] < x_train.loc[:, QI].drop_duplicates().shape[0])
    assert (anon.loc[:, QI].value_counts().min() >= k)
    np.testing.assert_array_equal(anon.drop(QI, axis=1), x_train.drop(QI, axis=1))
    anonymized_slice = anon.loc[:, QI_slices[0]]
    assert ((np.sum(anonymized_slice, axis=1) == 1).all())
    assert ((np.max(anonymized_slice, axis=1) == 1).all())
    assert ((np.min(anonymized_slice, axis=1) == 0).all())


def test_anonymize_pytorch_multi_label_binary():
    class multi_label_binary_model(nn.Module):
        def __init__(self, num_labels, num_features):
            super(multi_label_binary_model, self).__init__()

            self.fc1 = nn.Sequential(
                nn.Linear(num_features, 256),
                nn.Tanh(), )

            self.classifier1 = nn.Linear(256, num_labels)

        def forward(self, x):
            return self.classifier1(self.fc1(x))
            # missing sigmoid on each output

    class FocalLoss(nn.Module):
        def __init__(self, gamma=2, alpha=0.5):
            super(FocalLoss, self).__init__()
            self.gamma = gamma
            self.alpha = alpha

        def forward(self, input, target):
            bce_loss = functional.binary_cross_entropy_with_logits(input, target, reduction='none')

            p = sigmoid(input)
            p = where(target >= 0.5, p, 1 - p)

            modulating_factor = (1 - p) ** self.gamma
            alpha = self.alpha * target + (1 - self.alpha) * (1 - target)
            focal_loss = alpha * modulating_factor * bce_loss

            return focal_loss.mean()

    (x_train, y_train), _ = get_iris_dataset_np()

    # make multi-label binary
    y_train = np.column_stack((y_train, y_train, y_train))
    y_train[y_train > 1] = 1

    model = multi_label_binary_model(3, 4)
    criterion = FocalLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.01)

    art_model = PyTorchClassifier(model=model,
                                  output_type=CLASSIFIER_MULTI_OUTPUT_BINARY_LOGITS,
                                  loss=criterion,
                                  optimizer=optimizer,
                                  input_shape=(24,),
                                  nb_classes=3)
    art_model.fit(PytorchData(x_train.astype(np.float32), y_train.astype(np.float32)), save_entire_model=False,
                  nb_epochs=10)
    pred = art_model.predict(PytorchData(x_train.astype(np.float32), y_train.astype(np.float32)))
    pred = expit(pred)
    pred[pred < 0.5] = 0
    pred[pred >= 0.5] = 1

    k = 10
    QI = [0, 2]
    anonymizer = Anonymize(k, QI, train_only_QI=True)
    anon = anonymizer.anonymize(ArrayDataset(x_train, pred))
    assert (len(np.unique(anon[:, QI], axis=0)) < len(np.unique(x_train[:, QI], axis=0)))
    _, counts_elements = np.unique(anon[:, QI], return_counts=True)
    assert (np.min(counts_elements) >= k)
    assert ((np.delete(anon, QI, axis=1) == np.delete(x_train, QI, axis=1)).all())


def test_errors():
    with pytest.raises(ValueError):
        Anonymize(1, [0, 2])
    with pytest.raises(ValueError):
        Anonymize(2, [])
    with pytest.raises(ValueError):
        Anonymize(2, None)
    anonymizer = Anonymize(10, [0, 2])
    (x_train, y_train), (x_test, y_test) = get_iris_dataset_np()
    with pytest.raises(ValueError):
        anonymizer.anonymize(dataset=ArrayDataset(x_train, y_test))
    (x_train, y_train), _ = get_adult_dataset_pd()
    with pytest.raises(ValueError):
        anonymizer.anonymize(dataset=ArrayDataset(x_train, y_test))
