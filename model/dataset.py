"""
This file contains classes to work with the Dataset
"""

from os import path
from typing import List, Dict
from typing import Tuple, Any, Optional, Union, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.dataloader import _collate_fn_t, _worker_init_fn_t

from io_ import download_zip, read_json, store_json, get_dataset_dir, get_2d_file, get_mean_std_file, FILES
from io_ import log, log_progress, read_means_stds
from model.hand import HandCollection
from settings import FREIHAND_INFO, DATA, PRC


class FreiHANDDownloader:
    """
    This class provides an interface to download FreiHAND dataset
        and extract it to specific directory
    """

    # DUNDERS

    def __str__(self) -> str:
        """
        :return: object as a string
        """
        return f"FreiHANDDownloader [{self.dataset_dir}]"

    def __repr__(self) -> str:
        """
        :return: object as a string
        """
        return str(self)

    # UTILS

    @property
    def is_downloaded(self) -> bool:
        """
        :return: if the dataset was downloaded yet
        """
        return path.exists(self.dataset_dir)

    @property
    def dataset_dir(self) -> str:
        """
        :return: path to dataset directory
        """
        return get_dataset_dir()

    # DOWNLOAD

    def download(self):
        """
        It downloads the dataset if not downloaded yet
        """

        if self.is_downloaded:
            log(info=f"Dataset is already downloaded at {self.dataset_dir}")
            return

        log(info="Downloading dataset ")
        download_zip(
            url=FREIHAND_INFO["url"],
            dir_=self.dataset_dir
        )


class FreiHAND2DConverter:
    """
    This class provides an interface to convert 3-dimension points to 2-dimension ones
        using camera information; the new points are stored locally
    """

    # DUNDERS

    def __str__(self) -> str:
        """
        :return: object as a string
        """
        return f"FreiHAND2DConverter [{self.file_2d}]"

    def __repr__(self) -> str:
        """
        :return: object as a string
        """
        return str(self)

    # UTILS

    @property
    def file_2d(self) -> str:
        """
        :return: path to 2-dimension points file
        """
        return get_2d_file()

    @property
    def is_converted(self) -> bool:
        """
        :return: if the dataset was downloaded yet
        """
        return path.exists(self.file_2d)

    # CONVERT

    def convert_2d(self):
        """
        Uses 3d points and camera to convert to 2d points
            store the information in a .json file
        """

        if self.is_converted:
            log(info="2-dimension points converted yet")
            return

        log(info="Converting points to 2-dimension")

        # Loading json
        dataset_dir = get_dataset_dir()
        xyz_fp = path.join(dataset_dir, FILES["3d"])
        cameras_fp = path.join(dataset_dir, FILES["camera"])

        xyz = np.array(read_json(xyz_fp))
        cameras = np.array(read_json(cameras_fp))

        # Computing orientation
        uv = np.array([np.matmul(camera, xyz_.T).T for xyz_, camera in zip(xyz, cameras)])

        # Computing 2 dimension points
        xy = [list(uv_[:, :2] / uv_[:, -1:]) for uv_ in uv]

        # Cast to list a float
        xy = [[[float(x), float(y)] for x, y in inner_list] for inner_list in xy]

        # Store information
        store_json(path_=get_2d_file(), obj=xy)


class FreiHANDSplit:
    """
    This class computes indexes for data split
        and it provides the mean and standard deviation of training set
    """

    def __init__(self, n: int, percentages: List[float]):
        """
        It uses as dataset the first n images
        :param n: how many data in the dataset
        :param percentages: percentages of train, validation and test for te split
        """

        # Sanity checks
        if len(percentages) != 3:
            raise Exception(f"Got {len(percentages)} percentages for split, but they must be 3 ")

        if abs(sum(percentages) - 1) > 1e-9:  # floating point precision
            raise Exception(f"Invalid distribution [{percentages}] ")

        train_prc, val_prc, test_prc = percentages

        # Computing lengths
        train_len: int = int(n * train_prc)
        val_len: int = int(n * val_prc)
        test_len: int = n - (train_len + val_len)  # ensures a round split

        # Getting indexes
        self._train_idx: Tuple[int, int] = (0, train_len)
        self._val_idx: Tuple[int, int] = (train_len, train_len + val_len)
        self._test_idx: Tuple[int, int] = (n - test_len, n)

    def __len__(self) -> int:
        """
        :return: total length of splits
        """
        return self.train_len + self.val_len + self.test_len

    def __str__(self) -> str:
        """
        :return: string representation for the object
        """
        return f"FreiHANDSplit[Train: {self.train_len}; Validation: {self.val_len}; Test: {self.test_len}]"

    def __repr__(self) -> str:
        """
        :return: string representation for the object
        """
        return str(self)

    @property
    def train_idx(self) -> Tuple[int, int]:
        """
        :return: training set index boundaries
        """
        return self._train_idx

    @property
    def val_idx(self) -> Tuple[int, int]:
        """
        :return: validation set index boundaries
        """
        return self._val_idx

    @property
    def test_idx(self) -> Tuple[int, int]:
        """
        :return: test set index boundaries
        """
        return self._test_idx

    @property
    def train_len(self) -> int:
        """
        :return: train length
        """
        return self._interval_len(ab=self.train_idx)

    @property
    def val_len(self) -> int:
        """
        :return: train length
        """
        return self._interval_len(ab=self.val_idx)

    @property
    def test_len(self) -> int:
        """
        :return: train length
        """
        return self._interval_len(ab=self.test_idx)

    @staticmethod
    def _interval_len(ab: Tuple[int, int]) -> int:
        """
        Return the length of an interval
        :param ab: interval
        :return: interval length
        """
        a, b = ab
        return b - a

    def training_mean_std(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes mean and standard deviation over the training set
            for every channel of min-max scaled images
        Store the values in a .json file
        :return: means and standard deviations for every channel
        """

        channel_sum = np.zeros(3)  # for the mean
        channel_sum_squared = np.zeros(3)  # for the standard deviation

        collection = HandCollection()

        a, b = self.train_idx

        for train_idx in range(a, b):
            log_progress(idx=train_idx, max_=self.train_len, ckp=500)

            img_arr = collection.get_hand(idx=train_idx).image_arr_mm  # we apply mix-max scaling

            channel_sum += np.sum(img_arr, axis=(0, 1))
            channel_sum_squared += np.sum(img_arr ** 2, axis=(0, 1))

        # Get the total number of pixels
        new_size = DATA["new_size"]
        total_pixels = new_size * new_size * self.train_len

        # Computing mean and standard deviation
        channel_mean = channel_sum / total_pixels
        channel_std = np.sqrt((channel_sum_squared / total_pixels) - channel_mean ** 2)

        mean_std = {
            "mean": list(channel_mean),
            "std": list(channel_std)
        }

        store_json(
            path_=get_mean_std_file(),
            obj=mean_std
        )

        return channel_mean, channel_std


class FreiHANDDataset(Dataset):
    """
    Class to load FreiHAND dataset
    """

    def __init__(self, set_type: str):
        """
        :param set_type: name of set (train, val or test)
        """

        self._set_type = set_type
        self._collection = HandCollection()
        self._means, self._stds = read_means_stds()

        split_names = list(PRC.keys())
        percentages = list(PRC.values())

        split = FreiHANDSplit(n=DATA["n_data"], percentages=percentages)

        if set_type == split_names[0]:
            ab = split.train_idx
        elif set_type == split_names[1]:
            ab = split.val_idx
        elif set_type == split_names[2]:
            ab = split.test_idx
        else:
            raise Exception(f"Invalid set name {set_type};"\
                            " choose one between {train; validation; set}")

        a, b = ab
        self._indexes: List[int] = list(range(a, b))

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
        return len(self._indexes)

    def __getitem__(self, idx) -> Dict[str, Any]:
        """
        Return given pair image - heatmaps
        :param idx: data index
        :return: pair data-item, labels
        """

        actual_idx = self._indexes[idx]

        hand = self._collection.get_hand(idx=actual_idx)

        X = hand.image_arr_z(means=self._means, stds=self._stds)
        X = np.transpose(X, (2, 0, 1))  # move channels at first level
        X = torch.from_numpy(X)

        y = torch.from_numpy(hand.heatmaps)

        return {
            "image": X,
            "heatmaps": y,
            "image_name": hand.idx,
        }

    @property
    def set_type(self) -> str:
        """
        :return:  dataset set type
        """
        return self._set_type


class FreiHANDDataLoader(DataLoader):

    """
    This class implements the DataLoader for FreiHand Dataset
    """

    def __init__(self, dataset: FreiHANDDataset, batch_size: Optional[int] = 1, shuffle: Optional[bool] = None,
                 sampler: Union[Sampler, Iterable, None] = None,
                 batch_sampler: Union[Sampler[Sequence], Iterable[Sequence], None] = None, num_workers: int = 0,
                 collate_fn: Optional[_collate_fn_t] = None, pin_memory: bool = False, drop_last: bool = False,
                 timeout: float = 0, worker_init_fn: Optional[_worker_init_fn_t] = None, multiprocessing_context=None,
                 generator=None, *, prefetch_factor: int = 2, persistent_workers: bool = False,
                 pin_memory_device: str = ""):
        """

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
