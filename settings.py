"""
This file contains global variables
"""
import os
from os import path
from typing import Dict, List, Tuple

import torch

""" --------------- LOGGING --------------- """

LOG: bool = True  # if to log logic operation
LOG_IO: bool = False  # if to log i/o operation


""" ------------------------------- DIRECTORIES ------------------------------- """

FREIHAND_DIR: str = "FreiHAND"  # directory where to store the dataset
IMAGES: str = path.join("training", "rgb")  # directory where images are located


""" --------------- FILE NAMES (including extensions) ---------------- """
FILE_3D: str = "training_xyz.json"  # info about the camera
FILE_CAMERA: str = "training_K.json"  # info about 3d points
FILE_2D: str = "training_xy.json"  # info about 2d points (generated)
FILE_MEAN_STD: str = "mean_std.json"  # info about training set means and standard deviations (generated)


""" --------------------------------------------- DATASET INFORMATION --------------------------------------------- """

FREIHAND_URL: str = "https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2.zip"  # dataset download url

# IMAGES
ORIGINAL_SIZE: int = 224  # original size of images
IMG_EXT: str = "jpg"  # images extension

RAW: int = 32560  # how many raw images
AUGMENTED = 3  # how many augmented version per raw image
TOT_IMG = RAW * (AUGMENTED + 1)  # total number of images (raw + augmented)

# FINGERS
NUM_KEYPOINTS = 21

THUMB: str = "thumb"
INDEX: str = "index"
MIDDLE: str = "middle"
RING: str = "ring"
LITTLE: str = "little"

FINGERS = [THUMB, INDEX, MIDDLE, RING, LITTLE]

# CONNECTIONS
LINES: Dict[str, List[Tuple[int, int]]] = {
    THUMB: [(0, 1), (1, 2), (2, 3), (3, 4)],
    INDEX: [(0, 5), (5, 6), (6, 7), (7, 8)],
    MIDDLE: [(0, 9), (9, 10), (10, 11), (11, 12)],
    RING: [(0, 13), (13, 14), (14, 15), (15, 16)],
    LITTLE: [(0, 17), (17, 18), (18, 19), (19, 20)],
}


""" --------------- DATASET PREPARATION --------------- """
DATA: int = RAW  # data used for the model

NEW_SIZE: int = 128  # new image size
SIGMA_BLUR: float = 3.0  # heatmap blur

TRAIN_PRC: float = 0.7985257985257985  # percentage of training set
VALIDATION_PRC: float = 0.15356265356265356  # percentage of validation set
TEST_PRC: float = 0.04791154791154791  # percentage of test set

PRC: List[float] = [TRAIN_PRC, VALIDATION_PRC, TEST_PRC]


""" -- TRAIN, VALIDATION and TEST -- """

TRAIN_NAME = "training"
VALIDATION_NAME = "validation"
TEST_NAME = "test"


""" -------------- DRAW STYLE -------------- """

# KEYPOINTS
POINT: str = "383838"  # keypoint color
RADIUS: float = 1.5  # keypoint radius

# CONNECTIONS
WIDTH: int = 2  # width of connections

COLORS: Dict[str, str] = {  # Connection colors
    THUMB: "008000",
    INDEX: "00FFFF",
    MIDDLE: "0000FF",
    RING: "FF00FF",
    LITTLE: "FF0000"
}

""" ------------------ ML MODEL ------------------ """

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NEURONS: int = 16
"""N_EPOCHS: int = 500
BATCH_SIZE: int = 10
BATCHES_PER_EPOCH: int = 5
BATCHES_PER_EPOCH_VAL: int = 2"""
