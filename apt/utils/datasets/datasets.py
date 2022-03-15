# !/usr/bin/env python
"""
The AI Privacy Toolbox (datasets).
Implementation of utility classes for dataset handling
"""

from abc import ABCMeta, abstractmethod
from typing import Callable, Collection, Any, Union, List, Optional

import tarfile
import os
import urllib.request
import numpy as np
import pandas as pd
import logging
from torch import Tensor

logger = logging.getLogger(__name__)


INPUT_DATA_ARRAY_TYPE = Union[np.ndarray, pd.DataFrame, List, Tensor]
OUTPUT_DATA_ARRAY_TYPE = np.ndarray
DATA_PANDAS_NUMPY_TYPE = Union[np.ndarray, pd.DataFrame]


class Dataset(metaclass=ABCMeta):
    """Base Abstract Class for Dataset"""

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def get_samples(self) -> Collection[Any]:
        """Return data samples"""
        pass

    @abstractmethod
    def get_labels(self) -> Collection[Any]:
        """Return labels"""
        pass


class StoredDataset(Dataset):
    """Abstract Class for Storable Dataset"""

    @abstractmethod
    def load_from_file(self, path: str):
        """Load dataset from file"""
        pass

    @abstractmethod
    def load(self, **kwargs):
        """Load dataset"""
        pass

    @staticmethod
    def download(url: str, dest_path: str, filename: str, unzip: bool = False) -> None:
        """
        Download the dataset from URL
        :param url: dataset URL, the dataset will be requested from this URL
        :param dest_path: local dataset destination path
        :param filename: local dataset filename
        :param unzip: flag whether or not perform extraction
        :return: None
        """
        file_path = os.path.join(dest_path, filename)

        if os.path.exists(file_path):
            logger.warning("Files already downloaded, skipping downloading")

        else:
            os.makedirs(dest_path, exist_ok=True)
            logger.info("Downloading the dataset...")
            urllib.request.urlretrieve(url, file_path)
            logger.info('Dataset Downloaded')

        if unzip:
            StoredDataset.extract_archive(zip_path=file_path, dest_path=dest_path, remove_archive=False)


    @staticmethod
    def extract_archive(zip_path: str, dest_path=None, remove_archive=False):
        """
        Extract dataset from archived file
        :param zip_path: path to archived file
        :param dest_path: directory path to uncompress the file to
        :param remove_archive: whether remove the archive file after uncompress (default False)
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
    def split_debug(datafile: str, dest_datafile: str, ratio: int, shuffle=True, delimiter=",", fmt=None) -> None:
        """
        Split the data and take only a part of it
        :param datafile: dataset file path
        :param dest_datafile: destination path for the partial dataset file
        :param ratio: part of the dataset to save
        :param shuffle: whether to shuffle the data or not (default True)
        :param delimiter: dataset delimiter (default ",")
        :param fmt: format for the correct data saving
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

        debug_data = data[:int(len(data) * ratio)]
        logger.info(f"Saving {ratio} of the data to {dest_datafile}")
        np.savetxt(dest_datafile, debug_data, delimiter=delimiter, fmt=fmt)


class ArrayDataset(Dataset):
    """Dataset that is based on x and y arrays (e.g., numpy/pandas/list...)"""

    def __init__(self, x: INPUT_DATA_ARRAY_TYPE, y: Optional[INPUT_DATA_ARRAY_TYPE] = None, **kwargs):
        """
        ArrayDataset constructor.
        :param x: collection of data samples
        :param y: collection of labels (optional)
        :param kwargs: dataset parameters
        """
        # convert to numpy
        if type(x) == np.ndarray:
            self._x = x
        elif type(x) == pd.DataFrame:
            self._x = x.to_numpy()
        elif isinstance(x, list):
            self._x = np.array(x)
        elif type(x) == Tensor:
            self._x = x.numpy()
        else:
            raise ValueError('Non supported type for x: ', type(x).__name__)

        self._y = None
        if y is not None:
            if type(y) == np.ndarray:
                self._y = y
            elif type(y) == pd.DataFrame:
                self._y = y.to_numpy()
            elif isinstance(y, list):
                self._y = np.array(y)
            elif type(y) == Tensor:
                self._y = y.numpy()
            else:
                raise ValueError('Non supported type for y: ', type(y).__name__)

        if y is not None and len(self._x) != len(self._y):
            raise ValueError('Non equivalent lengths of x and y')

    def get_samples(self) -> OUTPUT_DATA_ARRAY_TYPE:
        """Return data samples as numpy array"""
        return self._x

    def get_labels(self) -> OUTPUT_DATA_ARRAY_TYPE:
        """Return labels as numpy array"""
        return self._y


class DatasetFactory:
    """Factory class for dataset creation"""
    registry = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """
        Class method to register Dataset to the internal registry
        :param name: dataset name
        :return:
        """
        def inner_wrapper(wrapped_class: Dataset) -> Any:
            if name in cls.registry:
                logger.warning('Dataset %s already exists. Will replace it', name)
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
        :param kwargs: dataset parameters
        :return: An instance of the dataset that is created.
        """
        if name not in cls.registry:
            msg = f'Dataset {name} does not exist in the registry'
            logger.error(msg)
            raise ValueError(msg)

        exec_class = cls.registry[name]
        executor = exec_class(**kwargs)
        return executor


class Data:
    def __init__(self, train: Dataset = None, test: Dataset = None, **kwargs):
        """
        Data class constructor.
        The class stores train and test datasets.
        If neither of the datasets was provided,
        Both train and test datasets will be create using
        DatasetFactory to create a dataset instance
        """
        if train or test:
            self.train = train
            self.test = test
        else:
            self.train = DatasetFactory.create_dataset(train=True, **kwargs)
            self.test = DatasetFactory.create_dataset(train=False, **kwargs)

    def get_train_set(self) -> Dataset:
        """Return train DatasetBase"""
        return self.train

    def get_test_set(self) -> Dataset:
        """Return test DatasetBase"""
        return self.test

    def get_train_samples(self) -> Collection[Any]:
        """Return train set samples"""
        return self.train.get_samples()

    def get_train_labels(self) -> Collection[Any]:
        """Return train set labels"""
        return self.train.get_labels()

    def get_test_samples(self) -> Collection[Any]:
        """Return test set samples"""
        return self.test.get_samples()

    def get_test_labels(self) -> Collection[Any]:
        """Return test set labels"""
        return self.test.get_labels()
