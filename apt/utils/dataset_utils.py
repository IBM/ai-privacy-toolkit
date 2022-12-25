from sklearn import datasets, model_selection
import sklearn.preprocessing
import pandas as pd
import ssl
from os import path, mkdir
from six.moves.urllib.request import urlretrieve


def get_iris_dataset_np(test_set: float = 0.3):
    """
    Loads the Iris dataset from scikit-learn.

    :param test_set: Proportion of the data to use as validation split (value between 0 and 1). Default is 0.3
    :type test_set: float
    :return: Entire dataset and labels as numpy arrays. Returned as a tuple (x_train, y_train), (x_test, y_test)
    """
    return _load_iris(test_set)


def _load_iris(test_set_size: float = 0.3):
    iris = datasets.load_iris()
    data = iris.data
    labels = iris.target

    # Split training and test sets
    x_train, x_test, y_train, y_test = model_selection.train_test_split(data, labels, test_size=test_set_size,
                                                                        random_state=18, stratify=labels)

    return (x_train, y_train), (x_test, y_test)


def get_diabetes_dataset_np(test_set: float = 0.3):
    """
    Loads the Diabetes dataset from scikit-learn.

    :param test_set: Proportion of the data to use as validation split (value between 0 and 1). Default is 0.3
    :type test_set: float
    :return: Entire dataset and labels as numpy arrays. Returned as a tuple (x_train, y_train), (x_test, y_test)
    """
    return _load_diabetes(test_set)


def _load_diabetes(test_set_size: float = 0.3):
    diabetes = datasets.load_diabetes()
    data = diabetes.data
    labels = diabetes.target

    # Split training and test sets
    x_train, x_test, y_train, y_test = model_selection.train_test_split(data, labels, test_size=test_set_size,
                                                                        random_state=18)

    return (x_train, y_train), (x_test, y_test)


def get_german_credit_dataset_pd(test_set: float = 0.3):
    """
    Loads the UCI German credit dataset from `datasets/german` or downloads it from
    https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/ if necessary.

    :param test_set: Proportion of the data to use as validation split (value between 0 and 1). Default is 0.3
    :type test_set: float
    :return: Dataset and labels as pandas dataframes. Returned as a tuple (x_train, y_train), (x_test, y_test)
    """

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data'
    data_dir = 'datasets/german'
    data_file = 'datasets/german/data'

    if not path.exists(data_dir):
        mkdir(data_dir)

    ssl._create_default_https_context = ssl._create_unverified_context
    if not path.exists(data_file):
        urlretrieve(url, data_file)

    # load data
    features = ["Existing_checking_account", "Duration_in_month", "Credit_history", "Purpose", "Credit_amount",
                "Savings_account", "Present_employment_since", "Installment_rate", "Personal_status_sex", "debtors",
                "Present_residence", "Property", "Age", "Other_installment_plans", "Housing",
                "Number_of_existing_credits", "Job", "N_people_being_liable_provide_maintenance", "Telephone",
                "Foreign_worker", "label"]
    data = pd.read_csv(data_file, sep=" ", names=features, engine="python")
    # remove rows with missing label
    data = data.dropna(subset=["label"])
    _modify_german_dataset(data)

    # Split training and test sets
    stratified = sklearn.model_selection.StratifiedShuffleSplit(n_splits=1, test_size=test_set, random_state=18)
    for train_set, test_set in stratified.split(data, data["label"]):
        train = data.iloc[train_set]
        test = data.iloc[test_set]
    x_train = train.drop(["label"], axis=1)
    y_train = train.loc[:, "label"]
    x_test = test.drop(["label"], axis=1)
    y_test = test.loc[:, "label"]

    x_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    x_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    return (x_train, y_train), (x_test, y_test)


def _modify_german_dataset(data):

    def modify_Foreign_worker(value):
        if value == 'A201':
            return 1
        elif value == 'A202':
            return 0
        else:
            raise Exception('Bad value')

    def modify_Telephone(value):
        if value == 'A191':
            return 0
        elif value == 'A192':
            return 1
        else:
            raise Exception('Bad value')

    def modify_label(value):
        return value - 1

    data['Foreign_worker'] = data['Foreign_worker'].apply(modify_Foreign_worker)
    data['Telephone'] = data['Telephone'].apply(modify_Telephone)
    data['label'] = data['label'].apply(modify_label)


def get_adult_dataset_pd():
    """
    Loads the UCI Adult dataset from `datasets/adult` or downloads it from
    https://archive.ics.uci.edu/ml/machine-learning-databases/adult/ if necessary.

    :return: Dataset and labels as pandas dataframes. Returned as a tuple (x_train, y_train), (x_test, y_test)
    """
    features = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                'label']
    train_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    test_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
    data_dir = 'datasets/adult'
    train_file = 'datasets/adult/train'
    test_file = 'datasets/adult/test'

    if not path.exists(data_dir):
        mkdir(data_dir)

    ssl._create_default_https_context = ssl._create_unverified_context
    if not path.exists(train_file):
        urlretrieve(train_url, train_file)
    if not path.exists(test_file):
        urlretrieve(test_url, test_file)

    train = pd.read_csv(train_file, sep=', ', names=features, engine='python')
    test = pd.read_csv(test_file, sep=', ', names=features, engine='python')
    test = test.iloc[1:]

    train = _modify_adult_dataset(train)
    test = _modify_adult_dataset(test)

    x_train = train.drop(['label'], axis=1)
    y_train = train.loc[:, 'label']
    x_test = test.drop(['label'], axis=1)
    y_test = test.loc[:, 'label']

    return (x_train, y_train), (x_test, y_test)


def _modify_adult_dataset(data):
    def modify_label(value):
        if value == '<=50K.' or value == '<=50K':
            return 0
        elif value == '>50K.' or value == '>50K':
            return 1
        else:
            raise Exception('Bad label value')

    def modify_native_country(value):
        Euro_1 = ['Italy', 'Holand-Netherlands', 'Germany', 'France']
        Euro_2 = ['Yugoslavia', 'South', 'Portugal', 'Poland', 'Hungary', 'Greece']
        SE_Asia = ['Vietnam', 'Thailand', 'Philippines', 'Laos', 'Cambodia']
        UnitedStates = ['United-States']
        LatinAmerica = ['Trinadad&Tobago', 'Puerto-Rico', 'Outlying-US(Guam-USVI-etc)', 'Nicaragua', 'Mexico',
                        'Jamaica', 'Honduras', 'Haiti', 'Guatemala', 'Dominican-Republic']
        China = ['Taiwan', 'Hong', 'China']
        BritishCommonwealth = ['Scotland', 'Ireland', 'India', 'England', 'Canada']
        SouthAmerica = ['Peru', 'El-Salvador', 'Ecuador', 'Columbia']
        Other = ['Japan', 'Iran', 'Cuba']

        if value in Euro_1:
            return 'Euro_1'
        elif value in Euro_2:
            return 'Euro_2'
        elif value in SE_Asia:
            return 'SE_Asia'
        elif value in UnitedStates:
            return 'UnitedStates'
        elif value in LatinAmerica:
            return 'LatinAmerica'
        elif value in China:
            return 'China'
        elif value in BritishCommonwealth:
            return 'BritishCommonwealth'
        elif value in SouthAmerica:
            return 'SouthAmerica'
        elif value in Other:
            return 'Other'
        elif value == '?':
            return 'Unknown'
        else:
            raise Exception('Bad native country value')

    data['label'] = data['label'].apply(modify_label)
    data['native-country'] = data['native-country'].apply(modify_native_country)

    for col in ('age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'):
        try:
            data[col] = data[col].fillna(0)
        except KeyError:
            print('missing column ' + col)

    for col in ('workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country'):
        try:
            data[col] = data[col].fillna('NA')
        except KeyError:
            print('missing column ' + col)

    return data.drop(['fnlwgt', 'education'], axis=1)


def get_nursery_dataset_pd(raw: bool = True, test_set: float = 0.2, transform_social: bool = False):
    """
    Loads the UCI Nursery dataset from `datasets/nursery` or downloads it from
    https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/ if necessary.

    :param raw: `True` if no preprocessing should be applied to the data. Otherwise, categorical data is one-hot
                encoded and data is scaled using sklearn's StandardScaler.
    :type raw: boolean
    :param test_set: Proportion of the data to use as validation split. The value should be between 0 and 1. Default is
                     0.2
    :type test_set: float
    :param transform_social: If `True`, transforms the social feature to be binary for the purpose of attribute
                             inference. This is done by assigning the original value 'problematic' the new value 1, and
                             the other original values are assigned the new value 0.
    :type transform_social: boolean
    :return: Dataset and labels as pandas dataframes. Returned as a tuple (x_train, y_train), (x_test, y_test)
    """
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.data'
    data_dir = 'datasets/nursery'
    data_file = 'datasets/nursery/data'

    if not path.exists(data_dir):
        mkdir(data_dir)

    ssl._create_default_https_context = ssl._create_unverified_context
    if not path.exists(data_file):
        urlretrieve(url, data_file)

    # load data
    features = ["parents", "has_nurs", "form", "children", "housing", "finance", "social", "health", "label"]
    categorical_features = ["parents", "has_nurs", "form", "housing", "finance", "social", "health"]
    data = pd.read_csv(data_file, sep=",", names=features, engine="python")
    # remove rows with missing label or too sparse label
    data = data.dropna(subset=["label"])
    data.drop(data.loc[data["label"] == "recommend"].index, axis=0, inplace=True)

    # fill missing values
    data["children"] = data["children"].fillna(0)

    for col in ["parents", "has_nurs", "form", "housing", "finance", "social", "health"]:
        data[col] = data[col].fillna("other")

    # make categorical label
    def modify_label(value):  # 5 classes
        if value == "not_recom":
            return 0
        elif value == "very_recom":
            return 1
        elif value == "priority":
            return 2
        elif value == "spec_prior":
            return 3
        else:
            raise Exception("Bad label value: %s" % value)

    data["label"] = data["label"].apply(modify_label)
    data["children"] = data["children"].apply(lambda x: "4" if x == "more" else x)

    if transform_social:

        def modify_social(value):
            if value == "problematic":
                return 1
            else:
                return 0

        data["social"] = data["social"].apply(modify_social)
        categorical_features.remove("social")

    if not raw:
        # one-hot-encode categorical features
        features_to_remove = []
        for feature in categorical_features:
            all_values = data.loc[:, feature]
            values = list(all_values.unique())
            data[feature] = pd.Categorical(data.loc[:, feature], categories=values, ordered=False)
            one_hot_vector = pd.get_dummies(data[feature], prefix=feature)
            data = pd.concat([data, one_hot_vector], axis=1)
            features_to_remove.append(feature)
        data = data.drop(features_to_remove, axis=1)

        # normalize data
        label = data.loc[:, "label"]
        features = data.drop(["label"], axis=1)
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(features)
        scaled_features = pd.DataFrame(scaler.transform(features), columns=features.columns)
        data = pd.concat([label, scaled_features], axis=1, join="inner")

    # Split training and test sets
    stratified = sklearn.model_selection.StratifiedShuffleSplit(n_splits=1, test_size=test_set, random_state=18)
    for train_set, test_set in stratified.split(data, data["label"]):
        train = data.iloc[train_set]
        test = data.iloc[test_set]
    x_train = train.drop(["label"], axis=1)
    y_train = train.loc[:, "label"]
    x_test = test.drop(["label"], axis=1)
    y_test = test.loc[:, "label"]

    return (x_train, y_train), (x_test, y_test)
