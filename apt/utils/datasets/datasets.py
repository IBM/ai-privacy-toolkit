# !/usr/bin/env python
"""
The AI Privacy Toolbox (datasets).
Implementation of utility classes for dataset handling
"""

from abc import ABCMeta, abstractmethod
from typing import Callable, Collection, Any, Union, List, Optional, Type

import tarfile
import os
import urllib.request
import numpy as np
import pandas as pd
import logging
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


INPUT_DATA_ARRAY_TYPE = Union[np.ndarray, pd.DataFrame, List, Tensor]
OUTPUT_DATA_ARRAY_TYPE = np.ndarray
DATA_PANDAS_NUMPY_TYPE = Union[np.ndarray, pd.DataFrame]


def array2numpy(arr: INPUT_DATA_ARRAY_TYPE) -> OUTPUT_DATA_ARRAY_TYPE:

    """
    converts from INPUT_DATA_ARRAY_TYPE to numpy array
    """
    if type(arr) == np.ndarray:
        return arr
    if type(arr) == pd.DataFrame or type(arr) == pd.Series:
        return arr.to_numpy()
    if isinstance(arr, list):
        return np.array(arr)
    if type(arr) == Tensor:
        return arr.detach().cpu().numpy()

    raise ValueError("Non supported type: ", type(arr).__name__)


def array2torch_tensor(arr: INPUT_DATA_ARRAY_TYPE) -> Tensor:
    """
    converts from INPUT_DATA_ARRAY_TYPE to torch tensor array
    """
    if type(arr) == np.ndarray:
        return torch.from_numpy(arr)
    if type(arr) == pd.DataFrame or type(arr) == pd.Series:
        return torch.from_numpy(arr.to_numpy())
    if isinstance(arr, list):
        return torch.tensor(arr)
    if type(arr) == Tensor:
        return arr

    raise ValueError("Non supported type: ", type(arr).__name__)


class Dataset(metaclass=ABCMeta):
    """Base Abstract Class for Dataset"""

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def get_samples(self) -> Collection[Any]:
        """
        Return data samples

        :return: the data samples
        """
        raise NotImplementedError

    @abstractmethod
    def get_labels(self) -> Collection[Any]:
        """
        Return labels

        :return: the labels
        """
        raise NotImplementedError

    @abstractmethod
    def get_predictions(self) -> OUTPUT_DATA_ARRAY_TYPE:
        """
        Get predictions

        :return: predictions as numpy array
        """
        raise NotImplementedError


class StoredDataset(Dataset):
    """Abstract Class for a Dataset that can be downloaded from a URL and stored in a file"""

    @abstractmethod
    def load_from_file(self, path: str):
        """
        Load dataset from file

        :param path: the path to the file
        :type path: string
        :return: None
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, **kwargs):
        """
        Load dataset

        :return: None
        """
        raise NotImplementedError

    @staticmethod
    def download(url: str, dest_path: str, filename: str, unzip: Optional[bool] = False) -> None:
        """
        Download the dataset from URL

        :param url: dataset URL, the dataset will be requested from this URL
        :type url: string
        :param dest_path: local dataset destination path
        :type dest_path: string
        :param filename: local dataset filename
        :type filename: string
        :param unzip: flag whether or not perform extraction. Default is False.
        :type unzip: boolean, optional
        :return: None
        """
        file_path = os.path.join(dest_path, filename)

        if os.path.exists(file_path):
            logger.warning("Files already downloaded, skipping downloading")

        else:
            os.makedirs(dest_path, exist_ok=True)
            logger.info("Downloading the dataset...")
            urllib.request.urlretrieve(url, file_path)
            logger.info("Dataset Downloaded")

        if unzip:
            StoredDataset.extract_archive(zip_path=file_path, dest_path=dest_path, remove_archive=False)

    @staticmethod
    def extract_archive(zip_path: str, dest_path: Optional[str] = None, remove_archive: Optional[bool] = False):
        """
        Extract dataset from archived file

        :param zip_path: path to archived file
        :type zip_path: string
        :param dest_path: directory path to uncompress the file to
        :type dest_path: string, optional
        :param remove_archive: whether remove the archive file after uncompress. Default is False.
        :type remove_archive: boolean, optional
        :return: None
        """
        logger.info("Extracting the dataset...")
        tar = tarfile.open(zip_path)
        tar.extractall(path=dest_path)

        logger.info("Dataset was extracted to {}".format(dest_path))
        if remove_archive:
            logger.info("Removing a zip file")
            os.remove(zip_path)
        logger.info("Extracted the dataset")

    @staticmethod
    def split_debug(datafile: str, dest_datafile: str, ratio: int, shuffle: Optional[bool] = True,
                    delimiter: Optional[str] = ",", fmt: Optional[Union[str, list]] = None) -> None:
        """
        Split the data and take only a part of it

        :param datafile: dataset file path
        :type datafile: string
        :param dest_datafile: destination path for the partial dataset file
        :type dest_datafile: string
        :param ratio: part of the dataset to save
        :type ratio: int
        :param shuffle: whether to shuffle the data or not. Default is True.
        :type shuffle: boolean, optional
        :param delimiter: dataset delimiter. Default is ","
        :type delimiter: string, optional
        :param fmt: format for the correct data saving. As defined by numpy.savetxt(). Default is None.
        :type fmt: string or sequence of strings, optional
        :return: None
        """
        if os.path.isfile(dest_datafile):
            logger.info(f"The partial debug split already exists {dest_datafile}")
            return
        else:
            os.makedirs(os.path.dirname(dest_datafile), exist_ok=True)

        data = np.genfromtxt(datafile, delimiter=delimiter)
        if shuffle:
            logger.info("Shuffling data")
            np.random.shuffle(data)

        debug_data = data[: int(len(data) * ratio)]
        logger.info(f"Saving {ratio} of the data to {dest_datafile}")
        np.savetxt(dest_datafile, debug_data, delimiter=delimiter, fmt=fmt)


class ArrayDataset(Dataset):
    """
    Dataset that is based on x and y arrays (e.g., numpy/pandas/list...)

    :param x: collection of data samples
    :type x: numpy array or pandas DataFrame or list or pytorch Tensor
    :param y: collection of labels
    :type y: numpy array or pandas DataFrame or list or pytorch Tensor, optional
    :param feature_names: The feature names, in the order that they appear in the data
    :type feature_names: list of strings, optional
    """

    def __init__(self, x: INPUT_DATA_ARRAY_TYPE, y: Optional[INPUT_DATA_ARRAY_TYPE] = None,
                 features_names: Optional[list] = None, **kwargs):
        self.is_pandas = self.is_pandas = type(x) == pd.DataFrame or type(x) == pd.Series

        self.features_names = features_names
        self._y = array2numpy(y) if y is not None else None
        self._x = array2numpy(x)

        if self.is_pandas:
            if features_names and not np.array_equal(features_names, x.columns):
                raise ValueError("The supplied features are not the same as in the data features")
            self.features_names = x.columns.to_list()

        if self._y is not None and len(self._x) != len(self._y):
            raise ValueError("Non equivalent lengths of x and y")

    def get_samples(self) -> OUTPUT_DATA_ARRAY_TYPE:
        """
        Get data samples

        :return: data samples as numpy array
        """
        return self._x

    def get_labels(self) -> OUTPUT_DATA_ARRAY_TYPE:
        """
        Get labels

        :return: labels as numpy array
        """
        return self._y

    def get_predictions(self) -> OUTPUT_DATA_ARRAY_TYPE:
        """
        Get predictions

        :return: predictions as numpy array
        """
        return None


class DatasetWithPredictions(Dataset):
    """
    Dataset that is based on arrays (e.g., numpy/pandas/list...). Includes predictions from a model, and possibly also
    features and true labels.

    :param x: collection of data samples
    :type x: numpy array or pandas DataFrame or list or pytorch Tensor
    :param y: collection of labels
    :type y: numpy array or pandas DataFrame or list or pytorch Tensor, optional
    :param feature_names: The feature names, in the order that they appear in the data
    :type feature_names: list of strings, optional
    """

    def __init__(self, pred: INPUT_DATA_ARRAY_TYPE, x: Optional[INPUT_DATA_ARRAY_TYPE] = None,
                 y: Optional[INPUT_DATA_ARRAY_TYPE] = None, features_names: Optional[list] = None, **kwargs):
        self.is_pandas = False
        self.features_names = features_names
        self._pred = array2numpy(pred)
        self._y = array2numpy(y) if y is not None else None
        self._x = array2numpy(x) if x is not None else None
        if self.is_pandas and x is not None:
            if features_names and not np.array_equal(features_names, x.columns):
                raise ValueError("The supplied features are not the same as in the data features")
            self.features_names = x.columns.to_list()

        if self._y is not None and len(self._pred) != len(self._y):
            raise ValueError('Non equivalent lengths of pred and y')

        if self._x is not None and len(self._x) != len(self._pred):
            raise ValueError('Non equivalent lengths of x and pred')

    def get_samples(self) -> OUTPUT_DATA_ARRAY_TYPE:
        """
        Get data samples

        :return: data samples as numpy array
        """
        return self._x

    def get_labels(self) -> OUTPUT_DATA_ARRAY_TYPE:
        """
        Get labels

        :return: labels as numpy array
        """
        return self._y

    def get_predictions(self) -> OUTPUT_DATA_ARRAY_TYPE:
        """
        Get predictions

        :return: predictions as numpy array
        """
        return self._pred


class PytorchData(Dataset):
    """
    Dataset for pytorch models.

    :param x: collection of data samples
    :type x: numpy array or pandas DataFrame or list or pytorch Tensor
    :param y: collection of labels
    :type y: numpy array or pandas DataFrame or list or pytorch Tensor, optional
    """
    def __init__(self, x: INPUT_DATA_ARRAY_TYPE, y: Optional[INPUT_DATA_ARRAY_TYPE] = None, **kwargs):
        self._y = array2torch_tensor(y) if y is not None else None
        self._x = array2torch_tensor(x)

        self.is_pandas = type(x) == pd.DataFrame or type(x) == pd.Series

        if self.is_pandas:
            self.features_names = x.columns

        if self._y is not None and len(self._x) != len(self._y):
            raise ValueError("Non equivalent lengths of x and y")

        if self._y is not None:
            self.__getitem__ = self.get_item
        else:
            self.__getitem__ = self.get_sample_item

    def get_samples(self) -> OUTPUT_DATA_ARRAY_TYPE:
        """
        Get data samples.

        :return: samples as numpy array
        """
        return array2numpy(self._x)

    def get_labels(self) -> OUTPUT_DATA_ARRAY_TYPE:
        """
        Get labels.

        :return: labels as numpy array
        """
        return array2numpy(self._y) if self._y is not None else None

    def get_predictions(self) -> OUTPUT_DATA_ARRAY_TYPE:
        """
        Get predictions

        :return: predictions as numpy array
        """
        return None

    def get_sample_item(self, idx: int) -> Tensor:
        """
        Get the sample according to the given index

        :param idx: the index of the sample to return
        :type idx: int
        :return: the sample as a pytorch Tensor
        """
        return self._x[idx]

    def get_item(self, idx: int) -> Tensor:
        """
        Get the sample and label according to the given index

        :param idx: the index of the sample to return
        :type idx: int
        :return: the sample and label as pytorch Tensors. Returned as a tuple (sample, label)
        """
        sample, label = self._x[idx], self._y[idx]
        return sample, label

    def __len__(self):
        return len(self._x)


class DatasetFactory:
    """Factory class for dataset creation"""

    registry = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """
        Class method to register Dataset to the internal registry

        :param name: dataset name
        :type name: string
        :return: a Callable that returns the registered dataset class
        """

        def inner_wrapper(wrapped_class: Type[Dataset]) -> Any:
            if name in cls.registry:
                logger.warning("Dataset %s already exists. Will replace it", name)
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create_dataset(cls, name: str, **kwargs) -> Dataset:
        """
        Factory command to create dataset instance.

        This method gets the appropriate Dataset class from the registry
        and creates an instance of it, while passing in the parameters
        given in ``kwargs``.

        :param name: The name of the dataset to create.
        :type name: string
        :param kwargs: dataset parameters
        :type kwargs: keyword arguments as expected by the class
        :return: An instance of the dataset that is created.
        """
        if name not in cls.registry:
            msg = f"Dataset {name} does not exist in the registry"
            logger.error(msg)
            raise ValueError(msg)

        exec_class = cls.registry[name]
        executor = exec_class(**kwargs)
        return executor


class Data:
    """
    Class for storing train and test datasets.

    :param train: the training set
    :type train: `Dataset`
    :param test: the test set
    :type test: `Dataset`, optional
    """
    def __init__(self, train: Dataset = None, test: Optional[Dataset] = None, **kwargs):
        """
        Data class constructor.

        If neither of the datasets was provided, both train and test datasets will be created using `DatasetFactory`.
        """
        if train or test:
            self.train = train
            self.test = test
        else:
            self.train = DatasetFactory.create_dataset(train=True, **kwargs)
            self.test = DatasetFactory.create_dataset(train=False, **kwargs)

    def get_train_set(self) -> Dataset:
        """
        Get training set

        :return: training 'Dataset`
        """
        return self.train

    def get_test_set(self) -> Dataset:
        """
        Get test set

        :return: test 'Dataset`
        """
        return self.test

    def get_train_samples(self) -> Collection[Any]:
        """
        Get train set samples, or None if no training data provided

        :return: training samples
        """
        if self.train is None:
            return None
        return self.train.get_samples()

    def get_train_labels(self) -> Collection[Any]:
        """
        Get train set labels, or None if no training labels provided

        :return: training labels
        """
        if self.train is None:
            return None
        return self.train.get_labels()

    def get_train_predictions(self) -> Collection[Any]:
        """
        Get train set predictions, or None if no training predictions provided

        :return: training labels
        """
        if self.train is None:
            return None
        return self.train.get_predictions()

    def get_test_samples(self) -> Collection[Any]:
        """
        Get test set samples

        :return: test samples, or None if no test data provided
        """
        if self.test is None:
            return None
        return self.test.get_samples()

    def get_test_labels(self) -> Collection[Any]:
        """
        Get test set labels

        :return: test labels, or None if no test labels provided
        """
        if self.test is None:
            return None
        return self.test.get_labels()

    def get_test_predictions(self) -> Collection[Any]:
        """
        Get test set predictions, or None if no test predictions provided

        :return: test labels
        """
        if self.test is None:
            return None
        return self.test.get_predictions()
