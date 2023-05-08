""" Pytorch Model Wrapper"""
import os
import shutil
import logging

from typing import Optional, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from art.utils import check_and_transform_label_format
from apt.utils.datasets.datasets import PytorchData
from apt.utils.models import Model, ModelOutputType
from apt.utils.datasets import OUTPUT_DATA_ARRAY_TYPE
from art.estimators.classification.pytorch import PyTorchClassifier as ArtPyTorchClassifier


logger = logging.getLogger(__name__)


class PyTorchModel(Model):
    """
    Wrapper class for pytorch models.
    """


class PyTorchClassifierWrapper(ArtPyTorchClassifier):
    """
    Wrapper class for pytorch classifier model.
    Extension for Pytorch ART model
    """

    def get_step_correct(self, outputs, targets) -> int:
        """
        Get number of correctly classified labels.
        """
        if len(outputs) != len(targets):
            raise ValueError("outputs and targets should be the same length.")
        if self.nb_classes > 1:
            return int(torch.sum(torch.argmax(outputs, axis=-1) == targets).item())
        else:
            return int(torch.sum(torch.round(outputs, axis=-1) == targets).item())

    def _eval(self, loader: DataLoader):
        """
        Inner function for model evaluation.
        """
        self.model.eval()

        total_loss = 0
        correct = 0
        total = 0

        for inputs, targets in loader:
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)

            outputs = self.model(inputs)
            loss = self._loss(outputs, targets)
            total_loss += loss.item() * targets.size(0)
            total += targets.size(0)
            correct += self.get_step_correct(outputs, targets)

        return total_loss / total, float(correct) / total

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_validation: np.ndarray = None,
        y_validation: np.ndarray = None,
        batch_size: int = 128,
        nb_epochs: int = 10,
        save_checkpoints: bool = True,
        save_entire_model=True,
        path=os.getcwd(),
        **kwargs,
    ) -> None:
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Training data.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or index labels
                  of shape (nb_samples,).
        :param x_validation: Validation data (optional).
        :param y_validation: Target validation values (class labels) one-hot-encoded of shape
                            (nb_samples, nb_classes) or index labels of shape (nb_samples,) (optional).
        :param batch_size: Size of batches.
        :param nb_epochs: Number of epochs to use for training.
        :param save_checkpoints: Boolean, save checkpoints if True.
        :param save_entire_model: Boolean, save entire model if True, else save state dict.
        :param path: path for saving checkpoint.
        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently
                       supported for PyTorch and providing it takes no effect.
        """
        # Put the model in the training mode
        self._model.train()

        if self._optimizer is None:  # pragma: no cover
            raise ValueError("An optimizer is needed to train the model, but none for provided.")

        y = check_and_transform_label_format(y, self.nb_classes)
        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=True)
        # Check label shape
        y_preprocessed = self.reduce_labels(y_preprocessed)

        train_dataset = TensorDataset(torch.from_numpy(x_preprocessed), torch.from_numpy(y_preprocessed))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if x_validation is None or y_validation is None:
            val_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
            logger.info("Using train set for validation")
        else:
            y_val = check_and_transform_label_format(y_validation, self.nb_classes)
            x_val_preprocessed, y_val_preprocessed = self._apply_preprocessing(x_validation, y_val, fit=False)
            # Check label shape
            y_val_preprocessed = self.reduce_labels(y_val_preprocessed)
            val_dataset = TensorDataset(torch.from_numpy(x_val_preprocessed), torch.from_numpy(y_val_preprocessed))
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        # Start training
        for epoch in range(nb_epochs):
            tot_correct = 0
            total = 0
            best_acc = 0
            # Shuffle the examples

            # Train for one epoch
            for inputs, targets in train_loader:
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)
                # Zero the parameter gradients
                self._optimizer.zero_grad()

                # Perform prediction
                model_outputs = self._model(inputs)

                # Form the loss function
                loss = self._loss(model_outputs[-1], targets)

                loss.backward()

                self._optimizer.step()
                correct = self.get_step_correct(model_outputs[-1], targets)
                tot_correct += correct
                total += targets.shape[0]

            val_loss, val_acc = self._eval(val_loader)
            logger.info(f"Epoch{epoch + 1}/{nb_epochs} Val_Loss: {val_loss}, Val_Acc: {val_acc}")

            best_acc = max(val_acc, best_acc)
            if save_checkpoints:
                if save_entire_model:
                    self.save_checkpoint_model(is_best=best_acc <= val_acc, path=path)
                else:
                    self.save_checkpoint_state_dict(is_best=best_acc <= val_acc, path=path)

    def save_checkpoint_state_dict(self, is_best: bool, path=os.getcwd(), filename="latest.tar") -> None:
        """
        Saves checkpoint as latest.tar or best.tar.

        :param is_best: whether the model is the best achieved model
        :param path: path for saving checkpoint
        :param filename: checkpoint name
        :return: None
        """
        # add path
        checkpoint = os.path.join(path, "checkpoints")
        path = checkpoint
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, filename)
        state = dict()
        state["state_dict"] = self.model.state_dict()
        state["opt_state_dict"] = self.optimizer.state_dict()

        logger.info(f"Saving checkpoint state dictionary: {filepath}")
        torch.save(state, filepath)
        if is_best:
            shutil.copyfile(filepath, os.path.join(path, "model_best.tar"))
            logger.info(f"Saving best state dictionary checkpoint: {os.path.join(path, 'model_best.tar')}")

    def save_checkpoint_model(self, is_best: bool, path=os.getcwd(), filename="latest.tar") -> None:
        """
        Saves checkpoint as latest.tar or best.tar.

        :param is_best: whether the model is the best achieved model
        :param path: path for saving checkpoint
        :param filename: checkpoint name
        :return: None
        """
        checkpoint = os.path.join(path, "checkpoints")
        path = checkpoint
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, filename)
        logger.info(f"Saving checkpoint model : {filepath}")
        torch.save(self.model, filepath)
        if is_best:
            shutil.copyfile(filepath, os.path.join(path, "model_best.tar"))
            logger.info(f"Saving best checkpoint model: {os.path.join(path, 'model_best.tar')}")

    def load_checkpoint_state_dict_by_path(self, model_name: str, path: str = None):
        """
        Load model only based on the check point path.

        :param model_name: check point filename
        :param path: checkpoint path (default current work dir)
        :return: loaded model
        """
        if path is None:
            path = os.path.join(os.getcwd(), "checkpoints")

        filepath = os.path.join(path, model_name)
        if not os.path.exists(filepath):
            msg = f"Model file {filepath} not found"
            logger.error(msg)
            raise FileNotFoundError(msg)

        else:
            checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)

        if self._optimizer and "opt_state_dict" in checkpoint:
            self._optimizer.load_state_dict(checkpoint["opt_state_dict"])
        self.model.eval()

    def load_latest_state_dict_checkpoint(self):
        """
        Load model state dict only based on the check point path (latest.tar).

        :return: loaded model
        """
        self.load_checkpoint_state_dict_by_path("latest.tar")

    def load_best_state_dict_checkpoint(self):
        """
        Load model state dict only based on the check point path (model_best.tar).

        :return: loaded model
        """
        self.load_checkpoint_state_dict_by_path("model_best.tar")

    def load_checkpoint_model_by_path(self, model_name: str, path: str = None):
        """
        Load model only based on the check point path.

        :param model_name: check point filename
        :param path: checkpoint path (default current work dir)
        :return: loaded model
        """
        if path is None:
            path = os.path.join(os.getcwd(), "checkpoints")

        filepath = os.path.join(path, model_name)
        if not os.path.exists(filepath):
            msg = f"Model file {filepath} not found"
            logger.error(msg)
            raise FileNotFoundError(msg)

        else:
            self._model._model = torch.load(filepath, map_location=self.device)
            self.model.to(self.device)
            self.model.eval()

    def load_latest_model_checkpoint(self):
        """
        Load entire model only based on the check point path (latest.tar).

        :return: loaded model
        """
        self.load_checkpoint_model_by_path("latest.tar")

    def load_best_model_checkpoint(self):
        """
        Load entire model only based on the check point path (model_best.tar).

        :return: loaded model
        """
        self.load_checkpoint_model_by_path("model_best.tar")


class PyTorchClassifier(PyTorchModel):
    """
    Wrapper class for pytorch classification models.
    """

    def __init__(
        self,
        model: "torch.nn.Module",
        output_type: ModelOutputType,
        loss: "torch.nn.modules.loss._Loss",
        input_shape: Tuple[int, ...],
        nb_classes: int,
        optimizer: "torch.optim.Optimizer",
        black_box_access: Optional[bool] = True,
        unlimited_queries: Optional[bool] = True,
        **kwargs,
    ):
        """
        Initialization specifically for the PyTorch-based implementation.

        :param model: PyTorch model. The output of the model can be logits, probabilities or anything else. Logits
                      output should be preferred where possible to ensure attack efficiency.
        :param output_type: The type of output the model yields (vector/label only for classifiers,
                            value for regressors)
        :param loss: The loss function for which to compute gradients for training. The target label must be raw
                     categorical, i.e. not converted to one-hot encoding.
        :param input_shape: The shape of one input instance.
        :param optimizer: The optimizer used to train the classifier.
        :param black_box_access: Boolean describing the type of deployment of the model (when in production).
                                 Set to True if the model is only available via query (API) access, i.e.,
                                 only the outputs of the model are exposed, and False if the model internals
                                 are also available. Optional, Default is True.
        :param unlimited_queries: If black_box_access is True, this boolean indicates whether a user can perform
                                  unlimited queries to the model API or whether there is a limit to the number of
                                  queries that can be submitted. Optional, Default is True.
        """
        super().__init__(model, output_type, black_box_access, unlimited_queries, **kwargs)
        self._loss = loss
        self._optimizer = optimizer
        self._art_model = PyTorchClassifierWrapper(model, loss, input_shape, nb_classes, optimizer)

    @property
    def loss(self):
        """
        The pytorch model's loss function.

        :return: The pytorch model's loss function
        """
        return self._loss

    @property
    def optimizer(self):
        """
        The pytorch model's optimizer.

        :return: The pytorch model's optimizer
        """
        return self._optimizer

    def fit(
        self,
        train_data: PytorchData,
        validation_data: PytorchData = None,
        batch_size: int = 128,
        nb_epochs: int = 10,
        save_checkpoints: bool = True,
        save_entire_model=True,
        path=os.getcwd(),
        **kwargs,
    ) -> None:
        """
        Fit the model using the training data.

        :param train_data: Training data.
        :type train_data: `PytorchData`
        :param validation_data: Training data.
        :type train_data: `PytorchData`
        :param batch_size: Size of batches.
        :param nb_epochs: Number of epochs to use for training.
        :param save_checkpoints: Boolean, save checkpoints if True.
        :param save_entire_model: Boolean, save entire model if True, else save state dict.
        :param path: path for saving checkpoint.
        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently
                       supported for PyTorch and providing it takes no effect.
        """
        if validation_data is None:
            self._art_model.fit(
                x=train_data.get_samples(),
                y=train_data.get_labels().reshape(-1, 1),
                batch_size=batch_size,
                nb_epochs=nb_epochs,
                save_checkpoints=save_checkpoints,
                save_entire_model=save_entire_model,
                path=path,
                **kwargs,
            )
        else:
            self._art_model.fit(
                x=train_data.get_samples(),
                y=train_data.get_labels().reshape(-1, 1),
                x_validation=validation_data.get_samples(),
                y_validation=validation_data.get_labels().reshape(-1, 1),
                batch_size=batch_size,
                nb_epochs=nb_epochs,
                save_checkpoints=save_checkpoints,
                save_entire_model=save_entire_model,
                path=path,
                **kwargs,
            )

    def predict(self, x: PytorchData, **kwargs) -> OUTPUT_DATA_ARRAY_TYPE:
        """
        Perform predictions using the model for input `x`.

        :param x: Input samples.
        :type x: `np.ndarray` or `pandas.DataFrame`
        :return: Predictions from the model (class probabilities, if supported).
        """
        return self._art_model.predict(x.get_samples(), **kwargs)

    def score(self, test_data: PytorchData, **kwargs):
        """
        Score the model using test data.

        :param test_data: Test data.
        :type test_data: `PytorchData`
        :return: the score as float (between 0 and 1)
        """
        y = check_and_transform_label_format(test_data.get_labels(), self._art_model.nb_classes)
        predicted = self.predict(test_data)
        return np.count_nonzero(np.argmax(y, axis=1) == np.argmax(predicted, axis=1)) / predicted.shape[0]

    def load_checkpoint_state_dict_by_path(self, model_name: str, path: str = None):
        """
        Load model only based on the check point path.

        :param model_name: check point filename
        :param path: checkpoint path (default current work dir)
        :return: loaded model
        """
        self._art_model.load_checkpoint_state_dict_by_path(model_name, path)

    def load_latest_state_dict_checkpoint(self):
        """
        Load model state dict only based on the check point path (latest.tar).

        :return: loaded model
        """
        self._art_model.load_latest_state_dict_checkpoint()

    def load_best_state_dict_checkpoint(self):
        """
        Load model state dict only based on the check point path (model_best.tar).

        :return: loaded model
        """
        self._art_model.load_best_state_dict_checkpoint()

    def load_checkpoint_model_by_path(self, model_name: str, path: str = None):
        """
        Load model only based on the check point path.

        :param model_name: check point filename
        :param path: checkpoint path (default current work dir)
        :return: loaded model
        """
        self._art_model.load_checkpoint_model_by_path(model_name, path)

    def load_latest_model_checkpoint(self):
        """
        Load entire model only based on the check point path (latest.tar).

        :return: loaded model
        """
        self._art_model.load_latest_model_checkpoint()

    def load_best_model_checkpoint(self):
        """
        Load entire model only based on the check point path (model_best.tar).

        :return: loaded model
        """
        self._art_model.load_best_model_checkpoint()
