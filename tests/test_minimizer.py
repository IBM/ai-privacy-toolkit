import pytest
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from sklearn.datasets import load_boston
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from apt.minimization import GeneralizeToRepresentative
from sklearn.tree import DecisionTreeClassifier


@pytest.fixture
def data():
    return load_boston(return_X_y=True)


def test_minimizer_params(data):
    # Assume two features, age and height, and boolean label
    cells = [{"id": 1, "ranges": {"age": {"start": None, "end": 38}, "height": {"start": None, "end": 170}}, "label": 0,
              "representative": {"age": 26, "height": 149}},
             {"id": 2, "ranges": {"age": {"start": 39, "end": None}, "height": {"start": None, "end": 170}}, "label": 1,
              "representative": {"age": 58, "height": 163}},
             {"id": 3, "ranges": {"age": {"start": None, "end": 38}, "height": {"start": 171, "end": None}}, "label": 0,
              "representative": {"age": 31, "height": 184}},
             {"id": 4, "ranges": {"age": {"start": 39, "end": None}, "height": {"start": 171, "end": None}}, "label": 1,
              "representative": {"age": 45, "height": 176}}
             ]
    features = ['age', 'height']
    X = np.array([[23, 165],
                  [45, 158],
                  [18, 190]])
    print(X.dtype)
    y = [1, 1, 0]
    base_est = DecisionTreeClassifier()
    base_est.fit(X, y)

    gen = GeneralizeToRepresentative(base_est, features=features, cells=cells)
    gen.fit()
    transformed = gen.transform(X)
    print(transformed)


def test_minimizer_fit(data):
    features = ['age', 'height']
    X = np.array([[23, 165],
                  [45, 158],
                  [56, 123],
                  [67, 154],
                  [45, 149],
                  [42, 166],
                  [73, 172],
                  [94, 168],
                  [69, 175],
                  [24, 181],
                  [18, 190]])
    print(X)
    y = [1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0]
    base_est = DecisionTreeClassifier()
    base_est.fit(X, y)
    predictions = base_est.predict(X)

    gen = GeneralizeToRepresentative(base_est, features=features, target_accuracy=0.5)
    gen.fit(X, predictions)
    transformed = gen.transform(X)
    print(X)
    print(transformed)


def test_minimizer_fit_pandas(data):
    features = ['age', 'height', 'sex', 'ola']
    X = [[23, 165, 'f', 'aa'],
         [45, 158, 'f', 'aa'],
         [56, 123, 'f', 'bb'],
         [67, 154, 'm', 'aa'],
         [45, 149, 'f', 'bb'],
         [42, 166, 'm', 'bb'],
         [73, 172, 'm', 'bb'],
         [94, 168, 'f', 'aa'],
         [69, 175, 'm', 'aa'],
         [24, 181, 'm', 'bb'],
         [18, 190, 'm', 'bb']]
    y = [1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0]
    X = pd.DataFrame(X, columns=features)

    numeric_features = ["age", "height"]
    numeric_transformer = Pipeline(
        steps=[('imputer', SimpleImputer(strategy='constant', fill_value=0))]
    )

    categorical_features = ["sex", "ola"]
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    encoded = preprocessor.fit_transform(X)
    base_est = DecisionTreeClassifier()
    base_est.fit(encoded, y)
    predictions = base_est.predict(encoded)
    # Append classifier to preprocessing pipeline.
    # Now we have a full prediction pipeline.
    gen = GeneralizeToRepresentative(base_est, features=features, target_accuracy=0.5,
                                     categorical_features=categorical_features)
    gen.fit(X, predictions)
    transformed = gen.transform(X)
    print(X)
    print(transformed)


def test_minimizer_fit_QI(data):
    features = ['age', 'height', 'weight']
    X = np.array([[23, 165, 70],
                  [45, 158, 67],
                  [56, 123, 65],
                  [67, 154, 90],
                  [45, 149, 67],
                  [42, 166, 58],
                  [73, 172, 68],
                  [94, 168, 69],
                  [69, 175, 80],
                  [24, 181, 95],
                  [18, 190, 102]])
    print(X)
    y = [1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0]
    QI = ['age', 'weight']
    base_est = DecisionTreeClassifier()
    base_est.fit(X, y)
    predictions = base_est.predict(X)

    gen = GeneralizeToRepresentative(base_est, features=features, target_accuracy=0.5, quasi_identifiers=QI)
    gen.fit(X, predictions)
    transformed = gen.transform(X)
    print(X)
    print(transformed)


def test_minimizer_fit_pandas_QI(data):
    features = ['age', 'height', 'weight', 'sex', 'ola']
    X = [[23, 165, 65, 'f', 'aa'],
         [45, 158, 76, 'f', 'aa'],
         [56, 123, 78, 'f', 'bb'],
         [67, 154, 87, 'm', 'aa'],
         [45, 149, 45, 'f', 'bb'],
         [42, 166, 76, 'm', 'bb'],
         [73, 172, 85, 'm', 'bb'],
         [94, 168, 92, 'f', 'aa'],
         [69, 175, 95, 'm', 'aa'],
         [24, 181, 49, 'm', 'bb'],
         [18, 190, 69, 'm', 'bb']]

    y = [1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0]
    X = pd.DataFrame(X, columns=features)

    numeric_features = ["age", "height", "weight"]
    numeric_transformer = Pipeline(
        steps=[('imputer', SimpleImputer(strategy='constant', fill_value=0))]
    )

    categorical_features = ["sex", "ola"]
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    encoded = preprocessor.fit_transform(X)
    base_est = DecisionTreeClassifier()
    base_est.fit(encoded, y)
    predictions = base_est.predict(encoded)
    # Append classifier to preprocessing pipeline.
    # Now we have a full prediction pipeline.
    QI = ['age', 'weight', 'ola']
    gen = GeneralizeToRepresentative(base_est, features=features, target_accuracy=0.5,
                                     categorical_features=categorical_features, quasi_identifiers=QI)
    gen.fit(X, predictions)
    transformed = gen.transform(X)
    print(X)
    print(transformed)


