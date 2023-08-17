"""
Training process
----------------

This module contains the Trainer class responsible for managing the training process,
including data loading, model optimization, loss calculation, and monitoring.
"""

from os import path
from typing import Dict, List

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from io_ import log, create_directory, store_json, get_loss_file, log_io, get_model_dir, get_model_file


class Trainer:

    """
    Class that encapsulates the training process of a neural network model, handling data loading,
    model optimization, loss computation, and monitoring. It also employs early stopping to mitigate overfitting.

    Attributes:
    - CHECKPOINT_FREQUENCY: Frequency at which to save model checkpoints during training.
    - EARLY_STOPPING_EPOCHS: Number of consecutive epochs with non-decreasing validation loss to trigger early stopping.
    - EARLY_STOPPING_AVG: Number of previous epochs to average for detecting early stopping.
    - EARLY_STOPPING_PRECISION: Decimal places to round validation loss when checking for early stopping.
    - model: The neural network model to be trained.
    - criterion: Loss function for training.
    - train_dataloader: DataLoader for the training dataset.
    - val_dataloader: DataLoader for the validation dataset.
    - device: Hardware component to run the model.
    - epochs: Number of training epochs.
    - batches_per_epoch: Number of batches per training epoch.
    - batches_per_epoch_val: Number of batches per validation epoch.
    - model_name: Name of the model.
    - X_name: Item name for feature vector.
    - y_name: Item name for label.
    - optimizer: Optimizer for updating model parameters.
    - scheduler: Learning rate scheduler to adjust learning rates during training.
    - loss: Dictionary containing training and validation loss values.
    """

    _CHECKPOINT_FREQUENCY: int = 50
    _EARLY_STOPPING_EPOCHS: int = 5
    _EARLY_STOPPING_AVG: int = 5
    _EARLY_STOPPING_PRECISION: int = 4

    def __init__(self, model: nn.Module, criterion: nn.Module,
                 train_dataloader: DataLoader, val_dataloader: DataLoader,
                 config: Dict[str, any]):
        """
        Initialize the Trainer instance with necessary parameters.

        :param model: the neural network model to be trained.
        :param criterion: loss function for training.
        :param train_dataloader: DataLoader for the training dataset.
        :param val_dataloader: DataLoader for the validation dataset.
        :param config: configuration dictionary, it must contain:
            - `device`: Hardware component to run the model.
            - `epochs`: Number of training epochs.
            - `batches_per_epoch`: Number of batches per training epoch.
            - `batches_per_epoch_val`: Number of batches per validation epoch.
            - `learning_rate`: Model learning rate.
            - `model_name`: Name of the model.
            - `X_name`: Item name for feature vector.
            - `y_name`: Item name for label.
        """

        self._model = model
        self._criterion = criterion

        self._device = config["device"]

        self._epochs = config["epochs"]
        self._batches_per_epoch = config["batches_per_epoch"]
        self._batches_per_epoch_val = config["batches_per_epoch_val"]

        self._model_name = config["model_name"]
        self._X_name = config["X_name"]
        self._y_name = config["y_name"]

        self._train_dataloader: DataLoader = train_dataloader
        self._val_dataloader: DataLoader = val_dataloader

        self._optimizer: SGD = SGD(params=model.parameters(), lr=config["learning_rate"])
        self._scheduler: ReduceLROnPlateau = ReduceLROnPlateau(
            optimizer=self._optimizer, factor=0.5, patience=20, verbose=True, threshold=0.00001
        )

        self._loss = {"train": [], "val": []}

    def __str__(self) -> str:
        """
        Return string representation for Trainer object.
        :return: string representation for the object.
        """
        return f"Trainer [Epochs: {self._epochs}; Batches per Epoch: {self._batches_per_epoch}; " \
               f"Batches per Epoch Validation: {self._batches_per_epoch_val}]"

    def __repr__(self) -> str:
        """
        Return string representation for Trainer object.
        :return: string representation for the object.
        """
        return str(self)

    def train(self) -> nn.Module:
        """
         Perform the training loop for the neural network model.

        :return: The trained neural network model.
        """

        # Create directory to save models
        create_directory(path_=get_model_dir(model_name=self._model_name))

        # Global variables for early stopping
        min_val_loss: float = .0
        no_decrease_epochs: int = 0

        # Iterates through epochs
        for epoch in range(self._epochs):

            # Perform training and validation for the current epoch
            self._epoch_train()
            self._epoch_eval()

            # Log epoch information
            log(
                info=f"Epoch: {epoch + 1}/{self._epochs}, "\
                     f"Train Loss={np.round(self._loss['train'][-1], 10)}, "\
                     f"Val Loss={np.round(self._loss['val'][-1], 10)}"
            )

            # Update learning rate using the scheduler
            self._scheduler.step(self._loss["train"][-1])

            # Save model if checkpoint is reached
            if (epoch + 1) % self._CHECKPOINT_FREQUENCY == 0:

                log(info=f"Saving checkpoint model: epoch {epoch+1}")
                epoch_str = str(epoch + 1).zfill(len(str(self._epochs-1)))
                self._save_model(suffix=epoch_str)

            # Check for early stopping
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

        # Save final model and loss information
        log(info="Saving final model")
        self._save_model(suffix='final')
        store_json(path_=get_loss_file(model_name=self._model_name), obj=self.loss)

        return self._model

    def _epoch_train(self):
        """
        Perform an epoch over the training set using mini-batches.

        This method trains the neural network model for one epoch using the training data in mini-batches.
        It computes forward and backward passes, updates model parameters, and calculates the average loss
        for the epoch.
        """

        # Set the model to training mode
        self._model.train()

        running_loss = []

        # Loop over training batches
        for i, data in enumerate(self._train_dataloader, 0):

            # Extract input images and ground truth heatmaps
            inputs = data[self._X_name].to(self._device)
            labels = data[self._y_name].to(self._device)

            # Clear gradients from previous batches
            self._optimizer.zero_grad()

            # Forward pass: compute predicted heatmaps
            outputs = self._model(inputs)

            # Compute the loss:
            loss = self._criterion(outputs, labels)

            # Backward pass: compute gradients and update model parameters
            loss.backward()

            # update model using gradients:
            self._optimizer.step()

            running_loss.append(loss.item())

            # Check if the current batch limit has been reached
            if i == self._batches_per_epoch:
                epoch_loss = np.mean(running_loss)
                self._loss["train"].append(epoch_loss)
                break

    def _epoch_eval(self):
        """
        Perform an epoch over the validation set using mini-batches.

        This method evaluates the neural network model on the validation data for one epoch using mini-batches.
        It computes forward passes, calculates the validation loss,
        and stores the average validation loss for the epoch.
        """

        # Set the model to evaluation mode
        self._model.eval()
        running_loss = []

        # Disable gradient computation during validation
        with torch.no_grad():

            # Loop over validation batches
            for i, data in enumerate(self._train_dataloader, 0):

                # Extract input images and ground truth heatmaps
                inputs = data[self._X_name].to(self._device)
                labels = data[self._y_name].to(self._device)

                # Compute predicted heatmaps
                outputs = self._model(inputs)

                # Compute the loss on validation data
                loss = self._criterion(outputs, labels)

                running_loss.append(loss.item())

                # Check if the current batch limit has been reached
                if i == self._batches_per_epoch_val:
                    epoch_loss = np.mean(running_loss)
                    self._loss["val"].append(epoch_loss)
                    break

    def _save_model(self, suffix: str):
        """
        Save the current model's state dictionary with the given suffix.

        :param suffix: string to be added to the model filename to distinguish between different checkpoints.
        """

        fp = get_model_file(model_name=self._model_name, suffix=suffix)
        log_io(info=f"Saving {fp}")
        torch.save(
            obj=self._model.state_dict(),
            f=fp
        )

    @property
    def loss(self) -> Dict[str, List[float]]:
        """
        Return a dictionary containing training and validation loss history.

        :return: dictionary containing the history of training and validation loss.
        """

        return self._loss

    def plot_loss(self):
        """
        Plot the training and validation loss over epochs.
        """

        epochs = range(1, len(self.loss["train"]) + 1)

        plt.plot(epochs, self.loss["train"], label="Train Loss")
        plt.plot(epochs, self.loss["val"], label="Validation Loss")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Train and Validation Loss")
        plt.legend()

        plt.show()
