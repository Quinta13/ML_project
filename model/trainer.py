from typing import Dict, List

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from io_ import log_progress, log, create_directory, get_model_dir, get_model_file, store_json, get_loss_file
from settings import MODEL_CONFIG


class Trainer:

    """
    Class for managing the training process (data loading, model optimization, loss calculation, and monitoring)
        in particular monitor loss in the validation set an stop the training when overfitting happens
    """

    _CHECKPOINT_FREQUENCY: int = 100  # when to save model
    _EARLY_STOPPING_EPOCHS: int = 10  # number of non-decreasing loss on validation
    _EARLY_STOPPING_AVG: int = 10  # number of previous epochs to be averaged for early stopping
    _EARLY_STOPPING_PRECISION: int = 5  # decimal places to round the validation loss for early stopping

    def __init__(self, model: nn.Module, criterion: nn.Module,
                 train_dataloader: DataLoader, val_dataloader: DataLoader,
                 epochs: int, batches_per_epoch: int, batches_per_epoch_val: int):

        self._model = model
        self._criterion = criterion

        self._device = MODEL_CONFIG["device"]

        self._epochs = epochs
        self._batches_per_epoch = batches_per_epoch
        self._batches_per_epoch_val = batches_per_epoch_val

        self._train_dataloader: DataLoader = train_dataloader
        self._val_dataloader: DataLoader = val_dataloader

        self._optimizer: SGD = SGD(params=model.parameters(), lr=MODEL_CONFIG["learning_rate"])
        self._scheduler: ReduceLROnPlateau = ReduceLROnPlateau(
            optimizer=self._optimizer, factor=0.5, patience=20, verbose=True, threshold=0.00001
        )

        self._loss = {"train": [], "val": []}

    def __str__(self) -> str:
        """
        :return: string representation for the object
        """
        return f"Trainer [Epochs: {self._epochs}; Batches per Epoch: {self._batches_per_epoch}; " \
               f"Batches per Epoch Validation: {self._batches_per_epoch_val}]"

    def __repr__(self) -> str:
        """
        :return: string representation for the object
        """
        return str(self)

    def train(self) -> nn.Module:
        """
        Performs the training loop
        :return: final model
        """

        # Creating directory where to save models
        create_directory(path_=get_model_dir())

        # Global variables
        min_val_loss: float = .0
        no_decrease_epochs: int = 0

        # Iterates through epochs
        for epoch in range(self._epochs):

            # Epoch fro
            self._epoch_train()
            self._epoch_eval()
            log(
                info=f"Epoch: {epoch + 1}/{self._epochs}, " \
                     f"Train Loss={np.round(self._loss['train'][-1], 10)}, "\
                     f"Val Loss={np.round(self._loss['val'][-1], 10)}"
            )

            # Update learning rate
            self._scheduler.step(self._loss["train"][-1])

            # Save model if checkpoint was reached
            if (epoch + 1) % self._CHECKPOINT_FREQUENCY == 0:
                self._save_model(suffix=str(epoch + 1).zfill(3))

            # Early stopping
            if epoch < self._EARLY_STOPPING_AVG:
                min_val_loss = np.round(np.mean(self._loss["val"]), self._EARLY_STOPPING_PRECISION)
                no_decrease_epochs = 0
            else:
                val_loss = np.round(
                    np.mean(self._loss["val"][-self._EARLY_STOPPING_AVG:]),
                    self._EARLY_STOPPING_PRECISION
                )
                if val_loss >= min_val_loss:
                    no_decrease_epochs += 1
                else:
                    min_val_loss = val_loss
                    no_decrease_epochs = 0

            if no_decrease_epochs > self._EARLY_STOPPING_EPOCHS:
                log(info="Early Stopping")
                break

        # Save model and loss
        self._save_model(suffix='final')
        store_json(path_=get_loss_file(), obj=self.loss)

        return self._model

    def _epoch_train(self):
        """
        It performs and epoch over the training set over a batch
        """

        # set the training mode
        self._model.train()

        running_loss = []

        # loop on batches
        for i, data in enumerate(self._train_dataloader, 0):

            # extract vectors and labels
            inputs = data['image'].to(self._device)
            labels = data['heatmaps'].to(self._device)

            # clear gradients from previous batches
            self._optimizer.zero_grad()

            # forward pass:
            outputs = self._model(inputs)

            # compute the loss:
            loss = self._criterion(outputs, labels)

            # backward pass:
            loss.backward()

            # update model using gradients:
            self._optimizer.step()

            running_loss.append(loss.item())

            # batch limit check
            if i == self._batches_per_epoch:
                epoch_loss = np.mean(running_loss)
                self._loss["train"].append(epoch_loss)
                break

    def _epoch_eval(self):
        """
        It performs and epoch over the validation set with weights derived from the training
            it compute the loss on the validation set
        """

        self._model.eval()
        running_loss = []

        with torch.no_grad():
            for i, data in enumerate(self._train_dataloader, 0):

                # extract vectors and labels
                inputs = data['image'].to(self._device)
                labels = data['heatmaps'].to(self._device)

                # compute loss on the actual model
                outputs = self._model(inputs)
                loss = self._criterion(outputs, labels)

                running_loss.append(loss.item())

                # batch limit check
                if i == self._batches_per_epoch_val:
                    epoch_loss = np.mean(running_loss)
                    self._loss["val"].append(epoch_loss)
                    break

    def _save_model(self, suffix: str):
        """
        It saves the actual model with given suffix
        :param suffix: suffix to model file
        """

        torch.save(
            obj=self._model.state_dict(),
            f=get_model_file(suffix=suffix)
        )

    @property
    def loss(self) -> Dict[str, List[float]]:
        """
        :return: train and validation loss
        """
        return self._loss

    def plot_loss(self):
        """
        It plots loss of training and validation over the epochs
        """

        epochs = range(1, len(self.loss["train"]) + 1)

        plt.plot(epochs, self.loss["train"], label="Train Loss")
        plt.plot(epochs, self.loss["val"], label="Validation Loss")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Train and Validation Loss")
        plt.legend()

        plt.show()
