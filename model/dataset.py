"""

FreiHAND Dataset Handling
-------------------------

This module contains classes for working with the FreiHAND dataset
 (downloading, conversion, splitting, loading, ...) tailored for hand pose estimation tasks.

Classes:
- FreiHANDDownloader: Provides methods for downloading and extracting the FreiHAND dataset.
- FreiHAND2DConverter: Converts 3D hand pose points to 2D points using camera information.
- FreiHANDSplit: Computes indexes for dataset splitting and calculates training set statistics.
- FreiHANDDataset: Custom PyTorch dataset class for loading FreiHAND data samples.
- FreiHANDDataLoader: Specialized DataLoader for efficient batch loading of FreiHANDDataset.

"""


from os import path
from typing import List, Dict
from typing import Tuple, Any, Optional, Union, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.dataloader import _collate_fn_t, _worker_init_fn_t
from tqdm import tqdm

from io_ import download_zip, read_json, store_json, get_dataset_dir, get_2d_file, get_mean_std_file, FILES
from io_ import log
from model.hand import HandCollection
from settings import FREIHAND_INFO, DATA, PRC


class FreiHANDDownloader:
    """
    This class provides an interface to download the FreiHAND dataset
     from a specified URL and extract it to a designated directory.

    Attributes:
     - dataset_dir str: path to dataset directory.
    """

    # CONSTRUCTOR

    def __init__(self):
        """
        Initialize the FreiHANDDownloader instance.

        The `dataset_dir` attribute is set to the path of the directory
         where the dataset will be stored.
        """

        self._dataset_dir: str = get_dataset_dir()

    # REPRESENTATION

    def __str__(self) -> str:
        """
        Return a string representation of the FreiHANDDownloader object.

        :returns: string representation of the object.
        """

        return f"FreiHANDDownloader [Dir: {self._dataset_dir}; Downloaded: {self.is_downloaded}]"

    def __repr__(self) -> str:
        """
        Return a string representation of the FreiHANDDownloader object.

        :returns: string representation of the object.
        """

        return str(self)

    # DOWNLOAD

    @property
    def is_downloaded(self) -> bool:
        """
        Check if the FreiHAND dataset is already downloaded.

        :return: True if the dataset is downloaded, False otherwise.
        """

        return path.exists(self._dataset_dir)

    def download(self):
        """
        Download and extract the FreiHAND dataset if not already downloaded.
        """

        # Check if already downloaded
        if self.is_downloaded:
            log(info=f"Dataset is already downloaded at {self._dataset_dir}")
            return

        # Download
        log(info="Downloading dataset ")
        download_zip(
            url=FREIHAND_INFO["url"],
            dir_=self._dataset_dir
        )


class FreiHAND2DConverter:
    """
    This class provides an interface to convert 3D hand keypoints to 2D keypoints using camera information.
    The converted 2D keypoints are stored locally in a JSON file.

    Attributes:
     - file_2d: path to the 2D keypoints JSON file.
    """

    # CONSTRUCTOR

    def __init__(self):
        """
        Initialize the FreiHAND2DConverter instance.

        The `file_2d` attribute is set to the path of the JSON file where
         the converted 2D keypoints will be stored.
        """

        self._file_2d: str = get_2d_file()

    # REPRESENTATION

    def __str__(self) -> str:
        """
        Return a string representation of the FreiHAND2DConverter object.

        :returns: string representation of the object.
        """

        return f"FreiHAND2DConverter [File: {self._file_2d}; Converted: {self.is_converted}]"

    def __repr__(self) -> str:
        """
        Return a string representation of the FreiHAND2DConverter object.

        :returns: string representation of the object.
        """

        return str(self)

    # CONVERSION

    @property
    def is_converted(self) -> bool:
        """
        Check if the 3D-to-2D conversion is already performed and stored.

        :return: True if the conversion is performed and stored, False otherwise.
        """

        return path.exists(self._file_2d)

    # CONVERT

    def convert_2d(self):
        """
        Convert 3D hand keypoints to 2D keypoints using camera information and store them in a JSON file.
        """

        # Check if already converted
        if self.is_converted:
            log(info="3D-to-2D conversion is already performed")
            return

        # Performing conversion

        log(info="Converting points to 2-dimension")

        # Load 3D keypoints and camera information
        dataset_dir = get_dataset_dir()
        xyz_fp = path.join(dataset_dir, FREIHAND_INFO["3d"])
        cameras_fp = path.join(dataset_dir, FREIHAND_INFO["camera"])

        xyz = np.array(read_json(xyz_fp))
        cameras = np.array(read_json(cameras_fp))

        # Perform 3D-to-2D conversion using camera information
        uv = np.array([np.matmul(camera, xyz_.T).T for xyz_, camera in zip(xyz, cameras)])
        xy = [list(uv_[:, :2] / uv_[:, -1:]) for uv_ in uv]

        # Cast to list a float
        xy = [[[float(x), float(y)] for x, y in inner_list] for inner_list in xy]

        # Store the converted 2D keypoints
        store_json(path_=get_2d_file(), obj=xy)


class FreiHANDSplitter:
    """
    This class computes indexes for data splitting and provides
     the mean and standard deviation of the training set for normalization.

    Attributes:
     - train_bounds: index boundaries for the training set.
     - val_bounds: index boundaries for the validation set.
     - test_bounds: index boundaries for the test set.
    """

    # CONSTRUCTOR

    def __init__(self, n: int, percentages: List[float]):
        """
        Initialize the FreiHANDSplit instance.

        :param n: total number of data samples.
        :param percentages: percentages of train, validation, and test data.

        :raises: Exception if the sum of percentages is not approximately equal to 1
                  or if the number of percentages is not 3.
        """

        # heck validity of input percentages
        if len(percentages) != 3:

            raise Exception(f"Invalid number of percentages."
                            f"Expected 3 percentages for train, validation, and test, got {len(percentages)}")

        if abs(sum(percentages) - 1) > 1e-9:  # floating point precision
            raise Exception(f"Invalid distribution percentages: {percentages}. The sum should be approximately 1.")

        # Computing splut lengths
        train_prc, val_prc, test_prc = percentages
        train_len: int = int(n * train_prc)
        val_len: int = int(n * val_prc)
        test_len: int = n - (train_len + val_len)  # Ensures a round split

        # Set index boundaries for different sets
        self._train_bounds: Tuple[int, int] = (0, train_len)
        self._val_bounds: Tuple[int, int] = (train_len, train_len + val_len)
        self._test_bounds: Tuple[int, int] = (n - test_len, n)

    # REPRESENTATION

    def __len__(self) -> int:
        """
        Get the total number of samples across all sets.

        :return: total number of samples.
        """

        return self.train_len + self.val_len + self.test_len

    def __str__(self) -> str:
        """
        Return a string representation of the FreiHANDSplitter object.

        :returns: string representation of the object.
        """

        return f"FreiHANDSplitter[Train: {self.train_len}; Validation: {self.val_len}; Test: {self.test_len}]"

    def __repr__(self) -> str:
        """
        Return a string representation of the FreiHANDSplitter object.

        :returns: string representation of the object.
        """

        return str(self)

    # INTERVALS

    @staticmethod
    def _bound_to_interval(bounds: Tuple[int, int]):
        """
        Returns a list of indexes within the given bounds.

        :param bounds: index boundaries as a tuple (start, end).
        :return: list of indexes
        """

        a, b = bounds
        return list(range(a, b))

    @property
    def train_idx(self) -> List[int]:
        """
        Get the list of indexes representing the training set.

        :return: List of training set indexes
        """

        return self._bound_to_interval(bounds=self.train_bounds)

    @property
    def val_idx(self) -> List[int]:
        """
        Get the list of indexes representing the validation set.

        :return: List of validation set indexes
        """

        return self._bound_to_interval(bounds=self.val_bounds)

    @property
    def test_idx(self) -> List[int]:
        """
        Get the list of indexes representing the test set.

        :return: List of test set indexes
        """

        return self._bound_to_interval(bounds=self.test_bounds)

    @property
    def train_bounds(self) -> Tuple[int, int]:
        """
        Get the index boundaries for the training set.

        :returns: training set index boundaries as a tuple (start, end).
        """

        return self._train_bounds

    @property
    def val_bounds(self) -> Tuple[int, int]:
        """
        Get the index boundaries for the validation set.

        :returns: validation set index boundaries as a tuple (start, end).
        """

        return self._val_bounds

    @property
    def test_bounds(self) -> Tuple[int, int]:
        """
        Get the index boundaries for the test set.

        :returns: test set index boundaries as a tuple (start, end).
        """

        return self._test_bounds

    @property
    def train_len(self) -> int:
        """
        Get the number of samples in the training set.

        :returns: number of samples in the training set.
        """

        return len(self.train_idx)

    @property
    def val_len(self) -> int:
        """
        Get the number of samples in the validation set.

        :returns: number of samples in the validation set.
        """

        return len(self.val_idx)

    @property
    def test_len(self) -> int:
        """
        Get the number of samples in the test set.

        :returns: number of samples in the test set.
        """

        return len(self.test_idx)

    # TRAINING SET STATISTICS

    def training_mean_std(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mean and standard deviation over the training set for normalization.

        :returns: means and standard deviations for each channel.
        """

        # Initialize accumulators for mean and standard deviation calculations
        channel_sum = np.zeros(3)
        channel_sum_squared = np.zeros(3)

        collection = HandCollection()

        log(info="Converting points")

        for train_idx in tqdm(self.train_idx):

            img_arr = collection[train_idx].image_arr_mm
            channel_sum += np.sum(img_arr, axis=(0, 1))
            channel_sum_squared += np.sum(img_arr ** 2, axis=(0, 1))

        # Get the total number of pixels
        new_size = DATA["new_size"]
        total_pixels = new_size * new_size * self.train_len

        # Compute mean and standard deviation
        channel_mean = channel_sum / total_pixels
        channel_std = np.sqrt((channel_sum_squared / total_pixels) - channel_mean ** 2)

        # Store mean and standard deviation in a JSON file
        mean_std = {
            "mean": list(channel_mean),
            "std": list(channel_std)
        }

        store_json(path_=get_mean_std_file(), obj=mean_std)

        return channel_mean, channel_std


class FreiHANDDataset(Dataset):
    """
    This class represents the FreiHAND dataset and provides an interface
     to load different sets (train, validation, or test) of data samples.

    Attributes:
    - set_type: string representing the set to use (train, validation, or test).
    - collection: HandCollection instance to load single items.
    - indexes: list of indexes for the type of dataset.
    """

    # CONSTRUCTOR

    def __init__(self, set_type: str):
        """
        Initialize the FreiHANDDataset instance.

        :param set_type: name of set (train, val or test)
        """

        self._set_type: str = set_type
        self._collection: HandCollection = HandCollection()

        split_names = list(PRC.keys())
        percentages = list(PRC.values())

        split = FreiHANDSplitter(n=DATA["n_data"], percentages=percentages)

        self._indexes: List[int]

        if set_type == split_names[0]:
            self._indexes = split.train_idx
        elif set_type == split_names[1]:
            self._indexes = split.val_idx
        elif set_type == split_names[2]:
            self._indexes = split.test_idx
        else:
            raise Exception(f"Invalid set name {set_type}; "
                            "choose one between {train; validation; set}")

    # REPRESENTATION

    def __str__(self) -> str:
        """
        Return a string representation of the FreiHANDDataset object.

        :returns: string representation of the object.
        """

        return f"FreiHAND [{self.set_type.capitalize()} - {len(self)} items]"

    def __repr__(self) -> str:
        """
        Return a string representation of the FreiHANDDataset object.

        :returns: string representation of the object.
        """

        return str(self)

    # OVERRIDES

    def __len__(self) -> int:
        """
        Get the number of data samples in the dataset.

        :return: number of data samples.
        """
        return len(self._indexes)

    def __getitem__(self, idx) -> Dict[str, Any]:
        """
        Get a data sample and its labels based on the index.

        :param idx: index of the data sample.
        :return: item containing the data sample, heatmaps, and image name.
        """

        # Compute index for the collection
        actual_idx = self._indexes[idx]

        hand = self._collection[actual_idx]

        # Convert data to tensors
        X = hand.image_arr_z
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
        Get the type of dataset set.

        :return: type of dataset set (train, validation, or test).
        """

        return self._set_type


class FreiHANDDataLoader(DataLoader):

    """
    This class implements the DataLoader for FreiHand Dataset
    """

    # CONSTRUCTOR OVERRIDE

    def __init__(self, dataset: FreiHANDDataset, batch_size: Optional[int] = 1, shuffle: Optional[bool] = None,
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

    def __str__(self) -> str:
        """
        Return a string representation of the FreiHANDDataLoader object.

        :returns: string representation of the object.
        """

        return f"FreiHANDDataLoader [{self.dataset.set_type.capitalize()} - Batch size: {self.batch_size} - Length: {len(self)}]"

    def __repr__(self) -> str:
        """
        Return a string representation of the FreiHANDDataLoader object.

        :returns: string representation of the object.
        """

        return str(self)
