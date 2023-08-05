"""
This file contains classes to prepare dataset for Machine Learning model
"""
from os import path
import random
from typing import Tuple, List

import numpy as np

from io_ import get_dataset_dir, log, download_zip, get_training_2d, read_json, get_training_3d, get_training_camera, \
    store_json, create_directory, get_data_dir, get_data_files, store_npy
from model.hand import HandCollection
from settings import FREIHAND_URL, TOT_IMG, DATA


class FreiHANDDownloader:

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

    @property
    def dataset_dir(self) -> str:
        """
        :return: path to dataset directory
        """
        return get_dataset_dir()

    @property
    def is_downloaded(self) -> bool:
        """
        :return: if the dataset was downloaded yet
        """
        return path.exists(self.dataset_dir)

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

    @property
    def file_2d(self) -> str:
        """
        :return: path to 2-dimension points file
        """
        return get_training_2d()

    @property
    def is_converted(self) -> bool:
        """
        :return: if the dataset was downloaded yet
        """
        return path.exists(self.file_2d)

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
        xyz = np.array(read_json(get_training_3d()))
        cameras = np.array(read_json(get_training_camera()))

        # Computing orientation
        uv = np.array([np.matmul(camera, xyz_.T).T for xyz_, camera in zip(xyz, cameras)])

        # Computing 2 dimension points
        xy = [list(uv_[:, :2] / uv_[:, -1:]) for uv_ in uv]

        # Cast to list an float
        xy = [[[float(x), float(y)] for x, y in inner_list] for inner_list in xy]

        # Store information
        store_json(path_=self.file_2d, obj=xy)


class DataPreprocessing:

    def __init__(self, data: int, train_prc: float,
                 val_prc: float, test_prc: float):
        """
        :param data: length of data
        :param train_prc: training set percentage
        :param val_prc: validation set percentage
        :param test_prc: test set percentage
        """

        # Hand collection
        self._hands = HandCollection()

        # Percentage check
        if abs(train_prc + val_prc + test_prc - 1) > 1e-9:  # floating point precision
            raise Exception(f"Invalid distribution [{train_prc}; {val_prc}; {test_prc}] ")

        # Indexes
        self._train_indexes: List[int]
        self._val_indexes: List[int]
        self._test_indexes: List[int]
        self._train_indexes, self._val_indexes, self._test_indexes = \
            self._create_partitions(data=data, train_prc=train_prc, val_prc=val_prc, test_prc=test_prc)

        # Splits - initialized after split method invocation
        self._train_image: np.ndarray | None = None
        self._train_heatmaps: np.ndarray | None = None
        self._val_image: np.ndarray | None = None
        self._val_heatmaps: np.ndarray | None = None
        self._test_image: np.ndarray | None = None
        self._test_heatmaps: np.ndarray | None = None

        self._split: bool = False

    def __str__(self) -> str:
        """
        :return: string representation for the object
        """

        train_len, val_len, test_len = self.lens
        return f"DataPreprocessing [Train: {train_len}; Validation: {val_len}; Test: {test_len}]"

    def __repr__(self) -> str:
        """
        :return: string representation for the object
        """
        return str(self)

    @property
    def train(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        :return: training set
        """
        self._check_split()
        return self._train_image, self._train_heatmaps

    @property
    def validation(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        :return: training set
        """
        self._check_split()
        return self._val_image, self._val_heatmaps

    @property
    def test(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        :return: training set
        """
        self._check_split()
        return self._test_image, self._test_heatmaps

    @property
    def lens(self) -> Tuple[int, int, int]:
        """
        :return: training, validation and test set lengths
        """
        return len(self._train_indexes), len(self._val_indexes), len(self._test_indexes)

    @staticmethod
    def _create_partitions(data: int, train_prc: float,
                           val_prc: float, test_prc: float) -> Tuple[List[int], List[int], List[int]]:
        """
        Create random data partitions basing on given data and percentages
        :param data: length of data
        :param train_prc: training set percentage
        :param val_prc: validation set percentage
        :param test_prc: test set percentage
        """

        # All possible indexes
        all_indexes = [i for i in range(TOT_IMG)]

        # Randomly select only given number of images
        random.shuffle(all_indexes)
        data_indexes = all_indexes[:DATA]

        # Computing length of partitions
        train_len = int(data * train_prc)
        val_len = int(data * val_prc)
        test_len = int(data * test_prc)

        # Computing slices
        train_indexes = data_indexes[:train_len]
        val_indexes = data_indexes[train_len:(train_len + val_len)]
        test_indexes = data_indexes[-test_len:]

        return train_indexes, val_indexes, test_indexes

    def split(self):
        """
        Split data applying min-max scaling and normalization over channels
        """

        # Getting images and keypoints
        log(info="Loading Training set")
        train_img, train_hm = self._load_images_heatmaps(idxs=self._train_indexes)
        log(info="Loading Validation set")
        val_img, val_hm = self._load_images_heatmaps(idxs=self._val_indexes)
        log(info="Loading Test set")
        test_img, test_hm = self._load_images_heatmaps(idxs=self._test_indexes)

        # Normalization in [0, 1]
        log(info="Applying mix-max scaling")
        train_img = train_img.astype(np.float32) / 255.
        val_img = val_img.astype(np.float32) / 255.
        test_img = test_img.astype(np.float32) / 255.

        # Computing mean and stds of training set per channel
        mean_ch = np.mean(train_img, axis=(0, 1, 2))
        std_ch = np.std(train_img, axis=(0, 1, 2))

        # Normalize
        log(info="Applying normalization")
        train_img_nrm = self._standard_normalization(images=train_img, means=mean_ch, stds=std_ch)
        val_img_nrm = self._standard_normalization(images=val_img, means=mean_ch, stds=std_ch)
        test_img_nrm = self._standard_normalization(images=test_img, means=mean_ch, stds=std_ch)

        # ------ LABELS ------

        # Save results
        self._train_image = train_img_nrm
        self._train_heatmaps = train_hm
        self._val_image = val_img_nrm
        self._val_heatmaps = val_hm
        self._test_image = test_img_nrm
        self._test_heatmaps = test_hm

        # Set split flag on
        self._split = True

    @staticmethod
    def _standard_normalization(images: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
        """
        Apply normalization to images to separate channels using given values for mean and standard deviation
        :param images: array of images
        :param means: mean value for every channel
        :param stds: standard deviation for every channel
        :return: normalized images
        """

        normalized = []

        for image in images:
            # Separate the RGB channels
            r_channel, g_channel, b_channel = image[:, :, 0], image[:, :, 1], image[:, :, 2]

            # Standard normalize each channel using the provided means and stds
            r_channel = (r_channel - means[0]) / stds[0]
            g_channel = (g_channel - means[1]) / stds[1]
            b_channel = (b_channel - means[2]) / stds[2]

            # Merge the normalized channels back into an image
            normalized_array = np.stack((r_channel, g_channel, b_channel), axis=-1)

            normalized.append(normalized_array)

        return np.array(normalized)

    def _load_images_heatmaps(self, idxs: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load images as arrays and associated labels
        :param idxs: indexes to load
        :return: tuple images - heatmaps
        """

        hands = [self._hands.get_hand(idx=idx) for idx in idxs]

        images = np.array([np.array(hand.image) for hand in hands])

        keypoints = np.array([hand.heatmaps for hand in hands])

        return images, keypoints

    def save(self):
        """
        Save dataset split to disk
        """

        self._check_split()

        # Create directory
        create_directory(path_=get_data_dir())

        # Files
        datas = [self.train, self.validation, self.test]

        # Get file paths
        train_fp, val_fp, test_fp = get_data_files()
        fps = [train_fp, val_fp, test_fp]

        # Save
        log(info="Saving files")
        for data, fp in zip(datas, fps):
            data_x, data_y = data
            fp_x, fp_y = fp

            store_npy(path_=fp_x, arr=data_x)
            store_npy(path_=fp_y, arr=data_y)

    def _check_split(self):
        """
        Check if split was performed
        """
        if not self._split:
            raise Exception(f"Dataset was not split yet. Use `split()` method to perform it")
