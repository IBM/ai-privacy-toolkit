import logging
import os
import random
import shutil
from typing import Optional, Tuple, Dict

import numpy as np
from art.utils import check_and_transform_label_format, logger

from sklearn.preprocessing import OneHotEncoder

from apt.utils.models import Model, ModelOutputType
from apt.utils.datasets import Dataset, OUTPUT_DATA_ARRAY_TYPE

from art.estimators.classification.pytorch import PyTorchClassifier as ArtPyTorchClassifier
import torch


class PyTorchModel(Model):
    """
    Wrapper class for pytorch models.
    """


class PyTorchClassifierWrapper(ArtPyTorchClassifier):
    """
        Wrapper class for pytorch classifier model.
        """

    def get_step_correct(self, outputs, targets) -> int:
        """get number of correctly classified labels"""
        if len(outputs) != len(targets):
            raise ValueError("outputs and targets should be the same length.")
        counter = 0
        for i, o in enumerate(outputs):
            if o == targets[i]:
                counter += 1
        return counter

    def fit(self, x: np.ndarray, y: np.ndarray, batch_size: int = 128, nb_epochs: int = 10,
            save_checkpoints: bool = True, **kwargs) -> None:
        """
                Fit the classifier on the training set `(x, y)`.

                :param x: Training data.
                :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or index labels of
                          shape (nb_samples,).
                :param batch_size: Size of batches.
                :param nb_epochs: Number of epochs to use for training.
                :param save_checkpoints: Boolean, save checkpoints if True.
                :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported for PyTorch
                       and providing it takes no effect.
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

        num_batch = int(np.ceil(len(x_preprocessed) / float(batch_size)))
        ind = np.arange(len(x_preprocessed))

        # Start training
        for epoch in range(nb_epochs):
            tot_correct = 0
            total = 0
            val_acc = 0
            best_acc = 0
            # Shuffle the examples
            random.shuffle(ind)

            # Train for one epoch
            for m in range(num_batch):
                i_batch = torch.from_numpy(x_preprocessed[ind[m * batch_size: (m + 1) * batch_size]]).to(self._device)
                o_batch = torch.from_numpy(y_preprocessed[ind[m * batch_size: (m + 1) * batch_size]]).to(self._device)

                # Zero the parameter gradients
                self._optimizer.zero_grad()

                # Perform prediction
                model_outputs = self._model(i_batch)

                # Form the loss function
                loss = self._loss(model_outputs[-1], o_batch)  # lgtm [py/call-to-non-callable]

                loss.backward()

                self._optimizer.step()
                correct = self.get_step_correct(model_outputs, o_batch)
                tot_correct += correct
                total += o_batch.shape[0]
            train_acc = float(tot_correct) / total

            if save_checkpoints:
                additional_states = {'epoch': epoch + 1, 'acc': train_acc, 'best_acc': val_acc}
                self.save_checkpoint(is_best=best_acc <= val_acc, additional_states=additional_states)
            best_acc = max(val_acc, best_acc)

    def save_checkpoint(self, is_best: bool, additional_states: Dict = None,
                        filename="latest.tar") -> None:
        """
        Saves checkpoint as latest.tar or best.tar
        :param is_best: whether the model is the best achieved model
        :param additional_states:  additional parameters that will be saved with the model
        :param filename: checkpoint name
        :return: None
        """
        checkpoint = os.path.join(os.getcwd(), 'checkpoints')
        path = checkpoint
        filepath = os.path.join(path, filename)
        state = additional_states if additional_states else dict()
        state['state_dict'] = self.model.module.state_dict() \
            if isinstance(self.model, torch.nn.DataParallel) else self.model.state_dict()
        state['opt_state_dict'] = self.optimizer.state_dict()
        torch.save(state, filepath)
        logging.info("Saving {} model with validation acc {} and train acc {}".
                     format('best' if is_best else 'checkpoint', state['best_acc'], state['acc']))
        if is_best:
            shutil.copyfile(filepath, os.path.join(path, 'model_best.tar'))

    def load_checkpoint(self, model_name: str, path: str = None):
        """
        Load model only based on the check point path
        :param model_name: check point filename
        :param path: checkpoint path (default current work dir)
        :return: loaded model
        """
        if path is None:
            path = os.path.join(os.getcwd(), 'checkpoints')

        filepath = os.path.join(path, model_name)
        if not os.path.exists(filepath):
            msg = f"Model file {filepath} not found"
            logger.error(msg)
            raise FileNotFoundError(msg)

        else:
            checkpoint = torch.load(filepath)
        if isinstance(self._model, torch.nn.DataParallel):
            self._model.module.load_state_dict(checkpoint['state_dict'])
        else:
            self._model.load_state_dict(checkpoint['state_dict'])

        if self._optimizer and 'opt_state_dict' in checkpoint:
            self._optimizer.load_state_dict(checkpoint['opt_state_dict'])


class PyTorchClassifier(PyTorchModel):
    """
    Wrapper class for pytorch classification models.
    """

    def __init__(self, model: "torch.nn.Module", output_type: ModelOutputType, loss: "torch.nn.modules.loss._Loss",
                 input_shape: Tuple[int, ...], nb_classes: int, optimizer: "torch.optim.Optimizer",
                 black_box_access: Optional[bool] = True, unlimited_queries: Optional[bool] = True, **kwargs):
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
        self._art_model = PyTorchClassifierWrapper(model, loss, input_shape, nb_classes, optimizer)

    def fit(self, train_data: Dataset, **kwargs) -> None:
        """
        Fit the model using the training data.

        :param train_data: Training data.
        :type train_data: `Dataset`
        """
        encoder = OneHotEncoder(sparse=False)
        y_encoded = encoder.fit_transform(train_data.get_labels().reshape(-1, 1))
        self._art_model.fit(train_data.get_samples(), y_encoded, **kwargs)

    def predict(self, x: Dataset, **kwargs) -> OUTPUT_DATA_ARRAY_TYPE:
        """
        Perform predictions using the model for input `x`.

        :param x: Input samples.
        :type x: `np.ndarray` or `pandas.DataFrame`
        :return: Predictions from the model (class probabilities, if supported).
        """
        return self._art_model.predict(x.get_samples(), **kwargs)

    def score(self, test_data: Dataset, **kwargs):
        """
        Score the model using test data.
        :param test_data: Test data.
        :type train_data: `Dataset`
        :return: the score as float (between 0 and 1)
        """
        y = check_and_transform_label_format(test_data.get_labels(), self._art_model.nb_classes)
        predicted = self.predict(test_data)
        return np.count_nonzero(np.argmax(y, axis=1) == np.argmax(predicted, axis=1)) / predicted.shape[0]
