"""
This class contains classes for Machine Learning algorithm
"""
from typing import Tuple, Any, Optional, Union, Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.dataloader import _collate_fn_t, _worker_init_fn_t

from io_ import log, log_progress, read_means_stds
from model.hand import HandCollection
from model.prepare import FreiHANDSplit
from settings import TRAIN_NAME, VALIDATION_NAME, TEST_NAME, DEVICE, N_EPOCHS, BATCHES_PER_EPOCH, BATCHES_PER_EPOCH_VAL, \
    PRC, DATA


class FreiHANDDataset(Dataset):
    """
    Class to load FreiHAND dataset
    """

    def __init__(self, set_type=TRAIN_NAME):
        """

        :param set_type: name of set (training, validation or test)
        """

        self._set_type = set_type
        self._device = DEVICE
        self._collection = HandCollection()
        self._means, self._stds = read_means_stds()

        split = FreiHANDSplit(n=DATA, percentages=PRC)

        if set_type == TRAIN_NAME:
            ab = split.train_idx
        elif set_type == VALIDATION_NAME:
            ab = split.val_idx
        elif set_type == TEST_NAME:
            ab = split.test_idx
        else:
            raise Exception(f"Invalid set name {set_type};"\
                            " choose one between [{TRAIN_NAME}; {VALIDATION_NAME_NAME}; {TEST_NAME}]")

        self._a, self._b = ab

    def __str__(self) -> str:
        """
        :return: string representation for the object
        """
        return f"FreiHAND [{self.set_type.capitalize()} - {len(self)} items]"

    def __repr__(self) -> str:
        """
        :return: string representation for the object
        """
        return str(self)

    def __len__(self) -> int:
        """
        :return: data length
        """
        return self._b - self._a

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        """
        Return given pair image - heatmaps
        :param idx: data index
        :return: pair data-item, labels
        """

        hand = self._collection.get_hand(idx=idx)

        X = hand.image_arr_z(means=self._means, stds=self._stds)
        X = np.transpose(X, (2, 0, 1))  # move channels at first level
        X = torch.tensor(X)

        y = torch.tensor(hand.heatmaps)

        return X, y

    @property
    def set_type(self) -> str:
        """
        :return:  dataset set type
        """
        return self._set_type


class FreiHANDDataLoader(DataLoader):

    def __init__(self, dataset: FreiHANDDataset, batch_size: Optional[int] = 1, shuffle: Optional[bool] = None,
                 sampler: Union[Sampler, Iterable, None] = None,
                 batch_sampler: Union[Sampler[Sequence], Iterable[Sequence], None] = None, num_workers: int = 0,
                 collate_fn: Optional[_collate_fn_t] = None, pin_memory: bool = False, drop_last: bool = False,
                 timeout: float = 0, worker_init_fn: Optional[_worker_init_fn_t] = None, multiprocessing_context=None,
                 generator=None, *, prefetch_factor: int = 2, persistent_workers: bool = False,
                 pin_memory_device: str = ""):
        super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, pin_memory,
                         drop_last, timeout, worker_init_fn, multiprocessing_context, generator,
                         prefetch_factor=prefetch_factor, persistent_workers=persistent_workers,
                         pin_memory_device=pin_memory_device)

    def __str__(self) -> str:
        """
        :return: string representation for the object
        """
        return f"FreiHANDDataLoader [{self.dataset.set_type.capitalize()} - Batch size: {self.batch_size} - Length: {len(self)}]"

    def __repr__(self) -> str:
        """
        :return: string representation for the object
        """
        return str(self)


class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.loss = {"train": [], "val": []}
        self.epochs = N_EPOCHS
        self.batches_per_epoch = BATCHES_PER_EPOCH
        self.batches_per_epoch_val = BATCHES_PER_EPOCH_VAL
        self.device = DEVICE
        self.scheduler = scheduler
        self.checkpoint_frequency = 100
        self.early_stopping_epochs = 10
        self.early_stopping_avg = 10
        self.early_stopping_precision = 5

    def train(self, train_dataloader, val_dataloader):

        for epoch in range(self.epochs):
            self._epoch_train(train_dataloader)
            self._epoch_eval(val_dataloader)
            print(
                "Epoch: {}/{}, Train Loss={}, Val Loss={}".format(
                    epoch + 1,
                    self.epochs,
                    np.round(self.loss["train"][-1], 10),
                    np.round(self.loss["val"][-1], 10),
                )
            )

            # reducing LR if no improvement
            if self.scheduler is not None:
                self.scheduler.step(self.loss["train"][-1])

            # saving model
            if (epoch + 1) % self.checkpoint_frequency == 0:
                torch.save(
                    self.model.state_dict(), "model_{}".format(str(epoch + 1).zfill(3))
                )

            # early stopping
            if epoch < self.early_stopping_avg:
                min_val_loss = np.round(np.mean(self.loss["val"]), self.early_stopping_precision)
                no_decrease_epochs = 0

            else:
                val_loss = np.round(
                    np.mean(self.loss["val"][-self.early_stopping_avg:]),
                    self.early_stopping_precision
                )
                if val_loss >= min_val_loss:
                    no_decrease_epochs += 1
                else:
                    min_val_loss = val_loss
                    no_decrease_epochs = 0
                    # print('New min: ', min_val_loss)

            if no_decrease_epochs > self.early_stopping_epochs:
                print("Early Stopping")
                break

        torch.save(self.model.state_dict(), "model_final")
        return self.model

    def _epoch_train(self, dataloader):
        self.model.train()
        running_loss = []

        for i, data in enumerate(dataloader, 1):

            log_progress(idx=i, max_=len(dataloader), ckp=2)

            inputs = data[0].to(self.device)
            labels = data[1].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss.append(loss.item())

            if i == self.batches_per_epoch:

                epoch_loss = np.mean(running_loss)
                self.loss["train"].append(epoch_loss)
                break

    def _epoch_eval(self, dataloader):
        self.model.eval()
        running_loss = []

        with torch.no_grad():
            for i, data in enumerate(dataloader, 1):

                log_progress(idx=i, max_=len(dataloader), ckp=2)

                inputs = data[0].to(self.device)
                labels = data[1].to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss.append(loss.item())

                if i == self.batches_per_epoch_val:
                    epoch_loss = np.mean(running_loss)
                    self.loss["val"].append(epoch_loss)
                    break


MODEL_NEURONS = 16


class ConvBlock(nn.Module):
    def __init__(self, in_depth, out_depth):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.BatchNorm2d(in_depth),
            nn.Conv2d(in_depth, out_depth, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_depth),
            nn.Conv2d(out_depth, out_depth, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class ShallowUNet(nn.Module):
    """
    Implementation of UNet, slightly modified:
    - less downsampling blocks
    - less neurons in the layers
    - Batch Normalization added

    Link to paper on original UNet:
    https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv_down1 = ConvBlock(in_channel, MODEL_NEURONS)
        self.conv_down2 = ConvBlock(MODEL_NEURONS, MODEL_NEURONS * 2)
        self.conv_down3 = ConvBlock(MODEL_NEURONS * 2, MODEL_NEURONS * 4)
        self.conv_bottleneck = ConvBlock(MODEL_NEURONS * 4, MODEL_NEURONS * 8)

        self.maxpool = nn.MaxPool2d(2)
        self.upsamle = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.conv_up1 = ConvBlock(
            MODEL_NEURONS * 8 + MODEL_NEURONS * 4, MODEL_NEURONS * 4
        )
        self.conv_up2 = ConvBlock(
            MODEL_NEURONS * 4 + MODEL_NEURONS * 2, MODEL_NEURONS * 2
        )
        self.conv_up3 = ConvBlock(MODEL_NEURONS * 2 + MODEL_NEURONS, MODEL_NEURONS)

        self.conv_out = nn.Sequential(
            nn.Conv2d(MODEL_NEURONS, out_channel, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        conv_d1 = self.conv_down1(x)
        conv_d2 = self.conv_down2(self.maxpool(conv_d1))
        conv_d3 = self.conv_down3(self.maxpool(conv_d2))
        conv_b = self.conv_bottleneck(self.maxpool(conv_d3))

        conv_u1 = self.conv_up1(torch.cat([self.upsamle(conv_b), conv_d3], dim=1))
        conv_u2 = self.conv_up2(torch.cat([self.upsamle(conv_u1), conv_d2], dim=1))
        conv_u3 = self.conv_up3(torch.cat([self.upsamle(conv_u2), conv_d1], dim=1))

        out = self.conv_out(conv_u3)
        return out


class IoULoss(nn.Module):
    """
    Intersection over Union Loss.
    IoU = Area of Overlap / Area of Union
    IoU loss is modified to use for heatmaps.
    """

    def __init__(self):
        super(IoULoss, self).__init__()
        self.EPSILON = 1e-6

    def _op_sum(self, x):
        return x.sum(-1).sum(-1)

    def forward(self, y_pred, y_true):
        inter = self._op_sum(y_true * y_pred)
        union = (
                self._op_sum(y_true ** 2)
                + self._op_sum(y_pred ** 2)
                - self._op_sum(y_true * y_pred)
        )
        iou = (inter + self.EPSILON) / (union + self.EPSILON)
        iou = torch.mean(iou)
        return 1 - iou
