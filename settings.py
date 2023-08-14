"""

Global Configuration File
-------------------------

This file contains global variables in dictionary form for:
- general information about the dataset (url, file paths, data format, ...).
- settings for data preparation (items to use, split percentages, ...).
- settings for Machine Learning model hyperparameters (neurons, learning rate, batch size, ...).

"""

from __future__ import annotations

from os import path
from typing import Dict, Any

import torch

"""
Dataset information
- `url` (str): URL to the dataset.
- `images` (str): File path to the images within the dataset.
- `3d` (str): File containing 3D coordinates data.
- `camera` (str): File containing camera information.
- `size` (int): Size of images (both width and height).
- `ext` (str): File extension for images.
- `raw` (int): Number of raw data.
- `n_keypoints` (int): Number of keypoints in the dataset.
- `idx_digits` (int): Number of digits images' index.
"""


FREIHAND_INFO: Dict[str, str | int] = {
    "url": "https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2.zip",
    "images": path.join("training", "rgb"),
    "3d": "training_xyz.json",
    "camera": "training_K.json",
    "size": 224,
    "ext": "jpg",
    "raw": 32560,
    "n_keypoints": 21,
    "idx_digits": 8
}

"""
Data Preparation Settings
- `n_data` (int): Items to use with the model.
- `new_size` (int): Target dimension for images resize (both width and height).
- `sigma_blur` (str): Radius for heatmap blur.


Dataset Percentages
- `train` (float): Training set percentage.
- `val` (float): Validation set percentage.
- `test` (float): Test set percentage.

The three value must sum up to 1.0 as compliant with probability distribution
"""

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

"""
Model Configuration
- `device` (torch.device): Device to be used for computation (GPU if available, else CPU).
- `in_channels` (int): Number of input channels (e.g., 3 for RGB images).
- `out_channels` (int): Number of output channels (e.g., number of keypoints).
- `learning_rate` (float): Learning rate for the model training.
- `epochs` (int): Number of training epochs.
- `batch_size` (int): Batch size used during training.
- `batches_per_epoch` (int): Number of batches processed in each training epoch.
- `batches_per_epoch_val` (int): Number of batches processed in each validation epoch.
"""

MODEL_CONFIG: Dict[str, Any] = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "in_channels": 3,
    "out_channels": 21,
    "learning_rate": 0.1,
    "epochs": 1000,
    "batch_size": 48,
    "batches_per_epoch": 50,
    "batches_per_epoch_val": 20
}
