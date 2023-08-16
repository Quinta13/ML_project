from typing import Dict, Any, Optional, Union, Iterable, Sequence

import torch
from torch import nn
from torch.utils.data import Sampler
from torch.utils.data.dataloader import _collate_fn_t, _worker_init_fn_t


from model.dataset import FreiHANDDataset, FreiHANDDataLoader


class FreiHANDDatasetLeftHand(FreiHANDDataset):

    def __init__(self, set_type: str):
        super().__init__(set_type)

    def __getitem__(self, idx) -> Dict[str, Any]:

        item = super().__getitem__(idx)

        left_hand = idx % 2 == 0

        X = item["image"]
        if not left_hand:
            X = torch.flip(X, [2])

        label = 1 if left_hand else 0
        y = torch.tensor(data=[label])

        return {
            "image": X,
            "left": y,
            "image_name": item["image_name"],
        }


class FreiHANDLeftHandDataLoader(FreiHANDDataLoader):

    """
    This class implements the DataLoader for FreiHand Dataset
    """

    # CONSTRUCTOR OVERRIDE

    def __init__(self, dataset: FreiHANDDatasetLeftHand, batch_size: Optional[int] = 1, shuffle: Optional[bool] = None,
                 sampler: Union[Sampler, Iterable, None] = None,
                 batch_sampler: Union[Sampler[Sequence], Iterable[Sequence], None] = None, num_workers: int = 0,
                 collate_fn: Optional[_collate_fn_t] = None, pin_memory: bool = False, drop_last: bool = False,
                 timeout: float = 0, worker_init_fn: Optional[_worker_init_fn_t] = None, multiprocessing_context=None,
                 generator=None, *, prefetch_factor: int = 2, persistent_workers: bool = False,
                 pin_memory_device: str = ""):
        """
        Initialize an instance of FreiHANDDataLoader.
        It basically specializes input dataset to FreiHANDDataset type.

        :param dataset: dataset from which to load the data.
        :param batch_size: how many samples per batch to load
            (default: ``1``).
        :param shuffle: set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
        :param sampler: defines the strategy to draw
            samples from the dataset. Can be any ``Iterable`` with ``__len__``
            implemented. If specified, :attr:`shuffle` must not be specified.
        :param batch_sampler: like :attr:`sampler`, but
            returns a batch of indices at a time. Mutually exclusive with
            :attr:`batch_size`, :attr:`shuffle`, :attr:`sampler`,
            and :attr:`drop_last`.
        :param num_workers: how many subprocesses to use for data
            loading. ``0`` means that the data will be loaded in the main process.
            (default: ``0``)
        :param collate_fn: merges a list of samples to form a
            mini-batch of Tensor(s).  Used when using batched loading from a
            map-style dataset.
        :param pin_memory: If ``True``, the data loader will copy Tensors
            into device/CUDA pinned memory before returning them.  If your data elements
            are a custom type, or your :attr:`collate_fn` returns a batch that is a custom type,
            see the example below.
        :param drop_last: set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: ``False``)
        :param timeout: if positive, the timeout value for collecting a batch
            from workers. Should always be non-negative. (default: ``0``)
        :param worker_init_fn: If not ``None``, this will be called on each
            worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as
            input, after seeding and before data loading. (default: ``None``)
        :param generator: If not ``None``, this RNG will be used
            by RandomSampler to generate random indexes and multiprocessing to generate
            `base_seed` for workers. (default: ``None``)
        :param prefetch_factor: Number of batches loaded
            in advance by each worker. ``2`` means there will be a total of
            2 * num_workers batches prefetched across all workers. (default: ``2``)
        :param persistent_workers: If ``True``, the data loader will not shutdown
            the worker processes after a dataset has been consumed once. This allows to
            maintain the workers `Dataset` instances alive. (default: ``False``)
        :param pin_memory_device: the data loader will copy Tensors
            into device pinned memory before returning them if pin_memory is set to true.
        """
        super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, pin_memory,
                         drop_last, timeout, worker_init_fn, multiprocessing_context, generator,
                         prefetch_factor=prefetch_factor, persistent_workers=persistent_workers,
                         pin_memory_device=pin_memory_device)


class LeftRightHandClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2),  # Output layer for binary classification
        )

    def forward(self, x):

        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

