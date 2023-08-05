"""
This file contains global variables
"""
from os import path
from typing import Dict, List, Tuple

# Dataset download
FREIHAND_URL: str = "https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2.zip"
FREIHAND_DIR: str = "FreiHAND"

IMG_EXT: str = "jpg"
ORIGINAL_SIZE: int = 224
NEW_SIZE: int = 128

RAW = 32560
AUGMENTED = 3
TOT_IMG = RAW * (AUGMENTED + 1)

# Training
TRAINING: str = path.join("training", "rgb")
TRAINING_3D: str = "training_xyz.json"
TRAINING_2D: str = "training_xy.json"
TRAINING_CAMERA: str = "training_K.json"


# Train - Test - Validation

DATA = 100

TRAIN_PRC = .7
TEST_PRC = .1
VAL_PRC = .2

DATA_DIR = "data"

TRAIN_NAME = "training"
VAL_NAME = "val"
TEST_NAME = "test"

VECTOR = "images"
LABELS = "labels"

# Logging
LOG: bool = True
LOG_IO: bool = False

# Finger Names
THUMB: str = "thumb"
INDEX: str = "index"
MIDDLE: str = "middle"
RING: str = "ring"
LITTLE: str = "little"

FINGERS = [THUMB, INDEX, MIDDLE, RING, LITTLE]

# Finger colors
COLORS: Dict[str, str] = {
    THUMB: "ED8671",
    INDEX: "BCD196",
    MIDDLE: "D9759A",
    RING: "EEC56B",
    LITTLE: "C0B8C6"
}

# Finger connections
NUM_KEYPOINTS = 21

WIDTH = 3
SIGMA_BLUR = 1.0

LINES: Dict[str, List[Tuple[int, int]]] = {
    THUMB: [(0, 1), (1, 2), (2, 3), (3, 4)],
    INDEX: [(0, 5), (5, 6), (6, 7), (7, 8)],
    MIDDLE: [(0, 9), (9, 10), (10, 11), (11, 12)],
    RING: [(0, 13), (13, 14), (14, 15), (15, 16)],
    LITTLE: [(0, 17), (17, 18), (18, 19), (19, 20)],
}
