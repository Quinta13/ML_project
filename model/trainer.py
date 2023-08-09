import numpy as np
import torch
from matplotlib import pyplot as plt

from io_ import log_progress, log, create_directory, get_model_dir, get_model_file, store_json, get_loss_file
from settings import DEVICE


class Trainer:
    _CHECKPOINT_FREQUENCY = 100
    _EARLY_STOPPING_EPOCHS = 10
    _EARLY_STOPPING_AVG = 10
    _EARLY_STOPPING_PRECISION = 5

    def __init__(self, model, criterion, optimizer, epochs, batches_per_epoch, batches_per_epoch_val, scheduler=None):

        self._model = model
        self._criterion = criterion
        self._optimizer = optimizer
        self._scheduler = scheduler

        self._device = DEVICE

        self._epochs = epochs
        self._batches_per_epoch = batches_per_epoch
        self._batches_per_epoch_val = batches_per_epoch_val

        self._loss = {"train": [], "val": []}

    def __str__(self) -> str:
        return f"Trainer [Epochs: {self._epochs}; Batches per Epoch: {self._batches_per_epoch}; " \
               f"Batches per Epoch Validation: {self._batches_per_epoch_val}]"

    def __repr__(self) -> str:
        return str(self)

    def train(self, train_dataloader, val_dataloader):

        create_directory(path_=get_model_dir())

        for epoch in range(self._epochs):
            self._epoch_train(train_dataloader)
            self._epoch_eval(val_dataloader)
            log(
                info=f"Epoch: {epoch + 1}/{self._epochs}, " \
                     f"Train Loss={np.round(self._loss['train'][-1], 10)}, " \
                     f"Val Loss={np.round(self._loss['val'][-1], 10)}"
            )

            # reducing LR if no improvement
            if self._scheduler is not None:
                self._scheduler.step(self._loss["train"][-1])

            # saving model
            if (epoch + 1) % self._CHECKPOINT_FREQUENCY == 0:
                self._save_model(suffix=str(epoch + 1).zfill(3))

            # early stopping
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

        self._save_model(suffix='final')
        store_json(path_=get_loss_file(), obj=self.loss)

        return self._model

    def _epoch_train(self, dataloader):
        self._model.train()
        running_loss = []

        for i, data in enumerate(dataloader, 0):

            log_progress(idx=i, max_=len(dataloader), ckp=20)

            inputs = data['image'].to(self._device)
            labels = data['heatmaps'].to(self._device)

            self._optimizer.zero_grad()

            outputs = self._model(inputs)
            loss = self._criterion(outputs, labels)
            loss.backward()
            self._optimizer.step()

            running_loss.append(loss.item())

            if i == self._batches_per_epoch:
                epoch_loss = np.mean(running_loss)
                self._loss["train"].append(epoch_loss)
                break

    def _epoch_eval(self, dataloader):
        self._model.eval()
        running_loss = []

        with torch.no_grad():
            for i, data in enumerate(dataloader, 0):

                log_progress(idx=i, max_=len(dataloader), ckp=5)

                inputs = data['image'].to(self._device)
                labels = data['heatmaps'].to(self._device)

                outputs = self._model(inputs)
                loss = self._criterion(outputs, labels)

                running_loss.append(loss.item())

                if i == self._batches_per_epoch_val:
                    epoch_loss = np.mean(running_loss)
                    self._loss["val"].append(epoch_loss)
                    break

    def _save_model(self, suffix: str):

        torch.save(
            self._model.state_dict(),
            f=get_model_file(suffix=suffix)
        )

    @property
    def loss(self):
        return self._loss

    def plot_loss(self):

        epochs = range(1, len(self.loss["train"]) + 1)

        plt.plot(epochs, self.loss["train"], label="Train Loss")
        plt.plot(epochs, self.loss["val"], label="Validation Loss")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Train and Validation Loss")
        plt.legend()

        plt.show()
