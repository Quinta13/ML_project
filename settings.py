"""
This file contains global variables
"""

from __future__ import annotations

from typing import Dict, Any

import torch

# DATASET INFORMATION

FREIHAND_INFO: Dict[str, str | int] = {
    "url": "https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2.zip",
    "size": 224,
    "ext": "jpg",
    "raw": 32560,
    "n_keypoints": 21
}

# DATASET PREPARATION

DATA: Dict[str, int | float | str] = {
    "n_data": 32560,
    "new_size": 128,
    "sigma_blur": 3.0
}

PRC: Dict[str, float] = {
    "train": 0.7985257985257985,
    "val": 0.15356265356265356,
    "test": 0.04791154791154791
}

# ML MODEL

MODEL_CONFIG: Dict[str, Any] = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "in_channel": 3,
    "out_channel": 21,
    "learning_rate": 0.1,
    "epochs": 1000,
    "batch_size": 48,
    "batches_per_epoch": 50,
    "batches_per_epoch_val": 20
}
