"""
This file contains classes to prepare dataset for Machine Learning model
"""
from os import path
from typing import List, Tuple, Iterator, Dict

import numpy as np

from io_ import log, download_zip, read_json, store_json, get_dataset_dir, get_2d_file, get_mean_std_file, log_progress
from model.hand import HandCollection
from settings import FREIHAND_URL, FILE_3D, FILE_CAMERA, NEW_SIZE


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
            url=FREIHAND_URL,
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
        xyz_fp = path.join(dataset_dir, FILE_3D)
        cameras_fp = path.join(dataset_dir, FILE_CAMERA)

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
        total_pixels = NEW_SIZE * NEW_SIZE * self.train_len

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
