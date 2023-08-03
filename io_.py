"""
This file contains general purpose input/output function
"""
import json
import os
from typing import Dict, List

import requests
import zipfile
from os import path

from PIL import Image

from settings import FREIHAND_DIR, LOG, LOG_IO, TRAINING, IMG_EXT, TRAINING_3D, TRAINING_CAMERA, TRAINING_2D

""" LOG """


def log(info: str):
    """
    Log information if enabled from settings
    :param info: information to be logged
    """
    if LOG:
        print(info)


def log_io(info: str):
    """
    Log i/o information if enabled from settings
    :param info: information to be logged
    """
    if LOG_IO:
        print(info)


""" DIRECTORIES """


def get_root_dir() -> str:
    """
    :return: path to root directory
    """
    return os.getcwd()


def get_dataset_dir() -> str:
    """
    :return: path to dataset directory
    """
    return os.path.join(get_root_dir(), FREIHAND_DIR)


def get_training_dir() -> str:
    """
    :return: path to training directory
    """
    return os.path.join(get_dataset_dir(), TRAINING)


def get_test_dir() -> str:
    """
    :return: path to test directory
    """
    return os.path.join(get_dataset_dir(), TEST)


""" FILES """


def get_training_camera() -> str:
    """
    :return: path to training camera
    """
    return os.path.join(get_dataset_dir(), TRAINING_CAMERA)


def get_training_3d() -> str:
    """
    :return: path to training 3d points
    """
    return os.path.join(get_dataset_dir(), TRAINING_3D)


def get_training_2d() -> str:
    """
    :return: path to training 2d points
    """
    return os.path.join(get_dataset_dir(), TRAINING_2D)


""" DOWNLOAD """


def download_zip(url: str, dir_: str):
    """
    This file downloads and extracts a .zip file from url to a target directory
    :param url: url to download file
    :param dir_: directory to extract file
    """

    # Create download directory if it doesn't exist
    log_io(info=f"Creating directory {dir_} ")
    os.makedirs(dir_, exist_ok=True)

    # Download zip file
    log_io(info=f"Downloading file from {url} ")
    response = requests.get(url)
    zip_file_path = os.path.join(dir_, 'tmp.zip')  # file will be removed at the end of the function

    with open(zip_file_path, 'wb') as file:
        file.write(response.content)

    # Extract the contents of the zip file
    log_io(info=f"Extracting zip in {zip_file_path} ")
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(dir_)

    # Remove the downloaded zip file
    os.remove(zip_file_path)


""" FILES """


def read_json(path_: str) -> Dict | List:
    """
    Load dictionary from local .json file
    :param path_: path for .json file
    :return: object
    """

    log_io(info=f"Loading {path_} ")

    with open(path_) as json_file:
        return json.load(json_file)


def store_json(path_: str, obj: Dict | List):
    """
    Stores given object as a json file
    :param path_: path for .json file
    :param obj: object to be stored
    """

    json_string = json.dumps(obj=obj)

    log_io(info=f"Saving {path_} ")

    with open(path_, 'w') as json_file:
        json_file.write(json_string)


""" IMAGE """


def _read_image(path_: str) -> Image:
    """
    Read image from local file
    :param path_: path to image file
    :return: image
    """

    log_io(info=f"Reading image {path_}")

    return Image.open(fp=path_)


def read_image(idx: int) -> Image:
    """
    Read an image from the directory given its index
    :param idx: image index
    :param training: to load from training set (if false it load from test set)
    :return: image
    """

    file_ = f"{str(idx).zfill(8)}.{IMG_EXT}"
    dir_ = get_training_dir()

    img_path = path.join(dir_, file_)

    return _read_image(path_=img_path)

#%%
