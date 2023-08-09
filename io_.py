"""
This file contains general purpose input/output function
"""
import json
import os
import zipfile
from os import path
from typing import Dict, List, Tuple

import numpy as np
import requests
from PIL import Image

from settings import LOG, LOG_IO, IMG_EXT, FREIHAND_DIR, FILE_2D, IMAGES, FILE_MEAN_STD, MODEL, LOSS_FILE
from utlis import pad_idx

""" LOG """


def log(info: str):
    """
    Log information if enabled from settings
    :param info: information to be logged
    """
    if LOG:
        print(f"INFO: {info}")


def log_io(info: str):
    """
    Log i/o information if enabled from settings
    :param info: information to be logged
    """
    if LOG_IO:
        print(f"I/O: {info}")


def log_progress(idx: int, max_: int, ckp: int = 100):
    """
    It logs progress over iterations
    :param idx: current iteration
    :param max_: maximum number of iteration
    :param ckp: checkpoint when to log
    """
    if idx % ckp == 0:
        log(info=f"Progress: [{idx}/{max_}] - {idx * 100 / max_:.2f}%")


""" DIRECTORIES """


def create_directory(path_: str):
    """
    Create directory if it doesn't exist
    :param path_: directory path
    """
    if os.path.exists(path_):
        log_io(f"Directory {path_} already exists ")
    else:
        os.makedirs(path_)
        log_io(f"Created directory {path_}")


def get_root_dir() -> str:
    """
    :return: path to root directory
    """
    return str(path.abspath(path.join(__file__, "../")))


def get_dataset_dir() -> str:
    """
    :return: path to root directory
    """
    return path.join(get_root_dir(), FREIHAND_DIR)


def get_images_dir() -> str:
    """
    :return: path to image directory
    """
    return path.join(get_dataset_dir(), IMAGES)


def get_model_dir() -> str:
    """
    :return: path to model directory
    """
    return path.join(get_dataset_dir(), MODEL)


def get_2d_file() -> str:
    """
    :return: path to 2-dimension file
    """
    return path.join(get_dataset_dir(), FILE_2D)


def get_mean_std_file() -> str:
    """
    :return: path to mean and standard deviation file
    """
    return path.join(get_dataset_dir(), FILE_MEAN_STD)


def get_loss_file() -> str:
    """
    :return: path to loss file
    """
    return path.join(get_model_dir(), LOSS_FILE)


def get_model_file(suffix: str = "final") -> str:
    """
    :param: suffix for the name of file indicative of the model
    :return: path to mean and standard deviation file
    """
    return path.join(get_model_dir(), f"{MODEL}_{suffix}")


def read_means_stds() -> Tuple[np.ndarray, np.ndarray]:
    """
    :return: means and standard deviations as arrays
    """
    means_stds = read_json(path_=get_mean_std_file())

    means, stds = means_stds.values()

    return np.array(means), np.array(stds)


""" DOWNLOAD """


def download_zip(url: str, dir_: str):
    """
    This file downloads and extracts a .zip file from url to a target directory
    :param url: url to download file
    :param dir_: directory to extract file
    """

    # Create download directory if it doesn't exist
    create_directory(path_=dir_)

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
        return json.load(fp=json_file)


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
    :return: image
    """

    file_ = f"{pad_idx(idx=idx)}.{IMG_EXT}"
    dir_ = get_images_dir()

    img_path = path.join(dir_, file_)

    return _read_image(path_=img_path)
