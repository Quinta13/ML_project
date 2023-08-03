"""
This file contains global variables
"""
from os import path
from typing import Dict, List, Tuple

# Dataset download
FREIHAND_URL: str = "https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2.zip"
FREIHAND_DIR: str = "FreiHAND"
IMG_EXT: str = "jpg"

# Training
TRAINING: str = path.join("training", "rgb")
TRAINING_3D: str = "training_xyz.json"
TRAINING_2D: str = "training_xy.json"
TRAINING_CAMERA: str = "training_K.json"

# Test
TEST: str = path.join("evaluation", "rgb")

# Logging
LOG: bool = True
LOG_IO: bool = True

# Finger Names
THUMB: str = "thumb"
INDEX: str = "index"
MIDDLE: str = "middle"
RING: str = "ring"
LITTLE: str = "little"

FINGERS = [THUMB, INDEX, MIDDLE, RING, LITTLE]

# Finger colors
COLORS: Dict[str, str] = {
    THUMB: "#ed8671",
    INDEX: "#bcd196",
    MIDDLE: "#d9759a",
    RING: "#eec56b",
    LITTLE: "#c0b8c6"
}

# Finger connections
CONNECTIONS: Dict[str, List[Tuple[int, int]]] = {
    THUMB: [(0, 1), (1, 2), (2, 3), (3, 4)],
    INDEX: [(0, 5), (5, 6), (6, 7), (7, 8)],
    MIDDLE: [(0, 9), (9, 10), (10, 11), (11, 12)],
    RING: [(0, 13), (13, 14), (14, 15), (15, 16)],
    LITTLE: [(0, 17), (17, 18), (18, 19), (20, 21)],
}
