"""
This file contains classes to prepare dataset for Machine Learning model
"""
from os import path

import numpy as np

from io_ import get_dataset_dir, log, download_zip, get_training_2d, read_json, get_training_3d, get_training_camera, \
    store_json
from settings import FREIHAND_URL


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
            log(info="Points yet converted yet")
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


