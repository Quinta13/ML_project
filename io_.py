"""
Input/Output Functions
----------------------

This module contains input/output functions for various tasks
 such as (logging, directory and files path handling, file operations, ...)

"""

from __future__ import annotations

import json
import os
import zipfile
from os import path
from typing import Dict, List, Tuple

import numpy as np
import requests
import torch
from PIL import Image
from torch import nn

from model.network import HandPoseEstimationUNet
from settings import FREIHAND_INFO, MODEL_CONFIG

"""
Global Logging Configuration

This boolean flag controls whether general logging is enabled or disabled,
 Use this flag to control the level of general information logging in the module.
 
- LOG` (bool): Set to `True` to enable general logging, and `False` to disable it.
- `LOG_IO` (bool): Set to `True` to enable I/O specific logging, and `False` to disable it.

"""

LOG: bool = True
LOG_IO: bool = False


# DIRECTORIES AND FILES
DIR_NAMES: Dict[str, str] = {
    """
    Directory Names
    
    This dictionary defines common directory names used in the module for different purposes.
    
    Attributes:
    - `freihand` (str): Name of the directory containing the FreiHAND dataset.
    - `external` (str): Name of the directory containing images external to the dataset.
    - `model` (str): Name of the directory containing machine learning model-related files.
    
    """
    "freihand": "FreiHAND",
    "external": "external",
    "model": "model"
}


FILES: Dict[str, str] = {
    """
    File Names
    
    This dictionary defines common file names used in the module for different purposes.
    
    Attributes:
    - `2d` (str): Name of the file containing 2D coordinate data.
    - `file_mean_std` (str): Name of the file containing mean and standard deviation
                              of the Training Set over different channels.
    - `loss` (str): Name of the file containing loss trend over Training and Validation set.
    - `model` (str): Base name of the model-related files (suffixes can be added for different versions).
    - `errors` (str): Name of the file containing errors on the Test set.
    
    """
    
    "2d": "training_xy.json",
    "file_mean_std": "mean_std.json",
    "loss": "loss.json",
    "model": "model",
    "errors": "errors.json"
}

# LOGGING FUNCTIONS


def log(info: str):
    """
    Log the provided information if general logging is enabled.

    :param info: information to be logged.
    """
    if LOG:
        print(f"INFO: {info}")


def log_io(info: str):
    """
    Log the provided input/output information if I/O logging is enabled.

    :param info: information to be logged.
    """

    if LOG_IO:
        print(f"I/O: {info}")


# DIRECTORY FUNCTIONS

def create_directory(path_: str):
    """
    Create a directory at the specified path if it does not already exist.

    :param path_: Path to the directory to be created.
    """

    if os.path.exists(path_):
        # directory already exists
        log_io(f"Directory {path_} already exists ")
    else:
        os.makedirs(path_)
        # directory doesn't exist
        log_io(f"Created directory {path_}")


def get_root_dir() -> str:
    """
    Get the path to project root directory.

    :return: path to the root directory.
    """

    return str(path.abspath(path.join(__file__, "../")))


def get_dataset_dir() -> str:
    """
    Get dataset directory.

    :return: path to the dataset directory.
    """

    return path.join(get_root_dir(), DIR_NAMES["freihand"])


def get_images_dir() -> str:
    """
    Get images directory.

    :return: path to the image directory.
    """

    return path.join(get_dataset_dir(), FREIHAND_INFO["images"])


def get_external_images() -> str:
    """
    Get external images directory.

    :return: path to the external image directory.
    """

    return path.join(get_dataset_dir(), DIR_NAMES["external"])


def get_model_dir() -> str:
    """
    Get model information directory.

    :return: path to the model directory.
    """

    return path.join(get_dataset_dir(), DIR_NAMES["model"])


# FILES


def get_2d_file() -> str:
    """
    Get 2D coordinate file.

    :return: path to the 2D coordinate file.
    """

    return path.join(get_dataset_dir(), FILES["2d"])


def get_mean_std_file() -> str:
    """
    Get mean and standard deviation file.

    :return: path to mean and standard deviation file.
    """

    return path.join(get_dataset_dir(), FILES["file_mean_std"])


def get_loss_file() -> str:
    """
    Get loss information file.

    :return: path to loss file.
    """

    return path.join(get_model_dir(), FILES["loss"])


def get_model_file(suffix: str = "final") -> str:
    """
    Get the path to a model file with an optional suffix.

    :param: suffix indicating the version of the model file.
    :return: path to the model file.
    """

    return path.join(get_model_dir(), f"{FILES['model']}_{suffix}")


def get_errors_file() -> str:
    """
    Get errors file.

    :return: path to errors file.
    """

    return path.join(get_model_dir(), FILES["errors"])


# OPERATIONS

def download_zip(url: str, dir_: str):
    """
    Download a zip file from a given URL and extract its contents to a target directory.

    :param url: URL to download the zip file from.
    :param dir_: directory path to extract the contents of the zip file.
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


def read_json(path_: str) -> Dict | List:
    """
    Load a dictionary or list from a local JSON file.

    :param path_: path to the JSON file to be read.
    :return: loaded JSON object (dictionary or list).
    """

    log_io(info=f"Loading {path_} ")

    with open(path_) as json_file:
        return json.load(fp=json_file)


def store_json(path_: str, obj: Dict | List):
    """
    Store a dictionary or list as a JSON file at the specified path.

    :param path_: path for the JSON file (to be created or overwritten).
    :param obj: JSON object (dictionary or list) to be stored.
    """

    json_string = json.dumps(obj=obj)

    # Check if file already existed
    info_op = "Saving" if path.exists(path_) else "Overwriting"

    log_io(info=f"{info_op} {path_} ")

    with open(path_, 'w') as json_file:
        json_file.write(json_string)


def load_model(path_: str) -> nn.Module:
    """
    Load a PyTorch model from a saved checkpoint file.

    :param path_: path to the model checkpoint file.
    :return: loaded PyTorch model.
    """

    # Model definition
    model = HandPoseEstimationUNet(
        in_channel=MODEL_CONFIG["in_channels"],
        out_channel=MODEL_CONFIG["out_channels"]
    )

    # Load the model
    log_io(info=f"Loading model {path_}")
    model.load_state_dict(
        state_dict=torch.load(
            f=path_,
            map_location=MODEL_CONFIG["device"]
        )
    )
    model.eval()

    return model


def read_means_stds() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load mean and standard deviation values from a local JSON file.

    :return: tuple containing  means and standard deviations.
    """

    means_stds = read_json(path_=get_mean_std_file())

    means, stds = means_stds.values()

    return np.array(means), np.array(stds)


def _load_image(path_: str) -> Image:
    """
    load an image from a local file.

    :param path_: path to image file to be load.
    :return: loaded image as a PIL Image object.
    """

    log_io(info=f"Reading image {path_}")

    return Image.open(fp=path_)


def load_image(idx: int) -> Image:
    """
    Read and load an image from a directory images based on its index.

    :param idx: index of the images to be read.
    :return: loaded image as a PIL Image object.
    """

    # Computing file path
    padded_idx = str(idx).zfill(FREIHAND_INFO["idx_digits"])
    file_ = f"{padded_idx}.{FREIHAND_INFO['ext']}"

    img_path = path.join(get_images_dir(), file_)

    return _load_image(path_=img_path)


def load_external_image(file_name: str) -> Image:
    """
    Read and load an image from a external images with a specified file name.

    :param file_name: name of the images to be read (including extension)
    :return: loaded image as a PIL Image object.
    """

    # Computing file path
    img_path = path.join(get_external_images(), file_name)

    return _load_image(path_=img_path)
