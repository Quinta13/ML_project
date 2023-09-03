"""

Hand handling
-------------

This module for handling Hand images, keypoints, transformations, and heatmaps.
 It includes functionality to read, transform, visualize, and analyze Hand images and keypoints.
 The module contains the following classes:

Classes:
- Hand: Represents an image of a hand with keypoints and provides methods for transformations and analysis.
- HandCollection: Manages a collection of Hand objects with keypoints.

"""

from __future__ import annotations

from typing import List, Tuple, Dict

import numpy as np
import torch
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from torch import nn

from io_ import read_json, load_image, get_2d_file, read_means_stds
from settings import DATA, FREIHAND_INFO

"""
Keypoints Connections

A dictionary that defines the connections between keypoints for different fingers.
The indices correspond to the keypoints of a hand and are used to visualize the connections between keypoints.

Attributes:
- thumb: Connections for the thumb finger.
- index: Connections for the index finger.
- middle: Connections for the middle finger.
- ring: Connections for the ring finger.
- little: Connections for the little finger.
"""
KEYPOINTS_CONNECTIONS: Dict[str, List[Tuple[int, int]]] = {
    "thumb": [(0, 1), (1, 2), (2, 3), (3, 4)],
    "index": [(0, 5), (5, 6), (6, 7), (7, 8)],
    "middle": [(0, 9), (9, 10), (10, 11), (11, 12)],
    "ring": [(0, 13), (13, 14), (14, 15), (15, 16)],
    "little": [(0, 17), (17, 18), (18, 19), (19, 20)],
}

"""
Draw Style

The dictionary that defines the drawing style parameters for keypoints visualization.

Attributes:
- point_color (str): Hexadecimal color code (e.g., "383838") used for drawing keypoints.
- point_radius (float): Radius of the circle used to represent keypoints.
- line_width (int): Width of the lines used to connect keypoints.

"""
STYLE: Dict[str, int | float | str] = {
    "point_color": "383838",
    "point_radius": 1.5,
    "line_width": 1
}

"""
Draw colors

A dictionary that defines colors for different fingers in keypoints visualization.

Attributes:
- thumb: Hexadecimal color code for the thumb finger.
- index: Hexadecimal color code for the index finger.
- middle: Hexadecimal color code for the middle finger.
- ring: Hexadecimal color code for the ring finger.
- little: Hexadecimal color code for the little finger.
"""
COLORS: Dict[str, str] = {
    "thumb": "008000",
    "index": "00FFFF",
    "middle": "0000FF",
    "ring": "FF00FF",
    "little": "FF0000"
}


class Hand:
    """
    This class represents an instance of a hand with image, keypoints, and related operations.

    The Hand class encapsulates information about a hand image along with its keypoints.
     It provides methods to perform various operations on the hand image and keypoints,
     including Z-transformation, heatmap generation, and visualization.

    It offers the possibility to mirror the image along vertical ax.

    Attributes:
    - idx: The dataset index of the hand image.
    - image: The original hand image.
    - keypoints: List of keypoints represented as (x, y) coordinates.
    - mirrored: Flag that indicated is the hand is mirrored
    """

    # CONSTRUCTOR

    def __init__(self, idx: int | str, image: Image, keypoints: List, mirrored: bool = False):
        """
        Initialize a Hand instance with the given dataset index, image, and keypoints.

        This constructor initializes a Hand object with the provided parameters.
         It resizes the image to the specified new size, scales the keypoints accordingly,
         and stores the processed data.

        :param idx: dataset index of the hand image, or name of the file.
        :param image: original hand image.
        :param keypoints: list of keypoints as (x, y) coordinates.
        :param mirrored: True if the hand is mirrored with respect to the original one, False otherwise.
        """

        # Pad to the correct number of digits
        self._idx: str = f"{str(idx).zfill(FREIHAND_INFO['idx_digits'])}.{FREIHAND_INFO['ext']}" \
            if type(idx) == int else idx

        # Resize to NEW_SIZE
        new_size = DATA["new_size"]
        original_size = image.width

        self._image: Image = image.resize((new_size, new_size))
        self._keypoints: List[Tuple[float, float]] = [
            (float(kp_x * new_size / original_size),
             float(kp_y * new_size / original_size))
            for kp_x, kp_y in keypoints
        ]

        self._mirrored: bool = mirrored

    # REPRESENTATION

    def __str__(self) -> str:
        """
        Return a string representation of the Hand object.

        :returns: string representation of the object.
        """

        return f"Hand[{self.idx} [{self.image_info}]"

    def __repr__(self) -> str:
        """
        Return a string representation of the Hand object.

        :returns: string representation of the object.
        """

        return str(self)

    # INFO

    @property
    def is_raw(self) -> bool:
        """
        Get information about the image.

        :returns: string containing information about the image, including size and mode.
        """

        return int(self.idx[:-4]) < FREIHAND_INFO["raw"]

    @property
    def is_augmented(self) -> bool:
        """
        Check if the image is from the augmented dataset.

        :returns: True if the image is from the augmented dataset, False otherwise.
        """

        return not self.is_raw

    @property
    def idx(self) -> str:
        """
        Get the formatted dataset index of the hand image.

        :returns: formatted dataset index
        """

        return self._idx

    @property
    def image_info(self) -> str:
        """
        Get information about the image.

        :returns: string containing information about the image, including size and mode.
        """
        return f"Size: {self.image.size}, Mode: {self.image.mode}"

    # MIRROR

    @property
    def is_mirrored(self) -> bool:
        """
        Get if the image is mirrored.

        :returns: True if hand is mirrored, false otherwise.
        """

        return self._mirrored

    def mirror(self) -> Hand:
        """
        Get mirrored hand.

        :returns: mirrored hand (both image and keypoints).
        """

        mirrored_image = self.image.transpose(Image.FLIP_LEFT_RIGHT)
        mirrored_keypoints = [(self.image.width - x, y) for x, y in self.keypoints]

        return Hand(
            idx=self.idx,
            image=mirrored_image,
            keypoints=mirrored_keypoints,
            mirrored=not self.is_mirrored
        )

    # IMAGE

    @property
    def image(self) -> Image:
        """
        Get the original image.

        :return: original image.
        """

        return self._image

    @property
    def image_arr(self) -> np.ndarray:
        """
        Get the image as a NumPy array.

        :return: image as a NumPy array of shape (height, width, channels).
        """

        return np.array(self._image).astype(dtype=np.float32)

    @property
    def image_arr_mm(self) -> np.ndarray:
        """
        Get the min-max scaled image as a NumPy array.
        :return: min-max scaled image as a NumPy array of shape (height, width, channels).
        """
        return self.image_arr / 255

    @property
    def image_arr_z(self) -> np.ndarray:
        """
        Get the Z-transformed image as a NumPy array.
        :return: Z-transformed image as a NumPy array of shape (height, width, channels).
        """

        # Get means and standard deviations computed on Training Set
        means, stds = read_means_stds()

        # Split channels
        img = self.image_arr_mm
        r_channel, g_channel, b_channel = img[:, :, 0], img[:, :, 1], img[:, :, 2]

        # Z-transform each channel with means and stds
        r_channel = (r_channel - means[0]) / stds[0]
        g_channel = (g_channel - means[1]) / stds[1]
        b_channel = (b_channel - means[2]) / stds[2]

        # Merge back into an image
        img_z = np.stack((r_channel, g_channel, b_channel), axis=-1)

        return np.clip(img_z, 0.0, 1.0)

    # KEYPOINTS

    @property
    def keypoints(self) -> List[Tuple]:
        """
        Get the keypoints of the hand.

        :returns: A list of tuples representing the keypoints of the hand.
                  Each tuple contains the (x, y) coordinates of a keypoint.
        """

        return self._keypoints

    def _draw_skeleton(self, keypoints: List[Tuple[float, float]]) -> Image:
        """
        Draw the skeleton and keypoints on an image.

        :param keypoints: list of tuples representing the keypoints to draw.
        :returns: new image with the skeleton and keypoints drawn.
        """

        # Create a copy of the image to draw on
        new_img = self.image.copy()
        draw = ImageDraw.Draw(new_img)

        # Convert the point color from hex to RGB
        color_point = tuple(int(STYLE["point_color"][i:i + 2], 16) for i in (0, 2, 4)) + (0,)  # from hex to rgb

        # Draw circles for each keypoint
        for keypoint in keypoints:
            radius = STYLE["point_radius"]

            x, y = keypoint
            x0 = x - radius
            y0 = y - radius
            x1 = x + radius
            y1 = y + radius

            # Draw the circle with the specified color and alpha
            draw.ellipse([(x0, y0), (x1, y1)], fill=color_point)

        # Draw lines connecting keypoints to represent fingers
        for finger_key, connection in KEYPOINTS_CONNECTIONS.items():

            # Finger color
            color = COLORS[finger_key]
            color_rgb = tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))  # from hex to rgb

            # Finger connections
            for line in connection:
                p1, p2 = line
                draw.line([keypoints[p1], keypoints[p2]], fill=color_rgb, width=STYLE["line_width"])

        return new_img

    @property
    def skeleton(self) -> Image:
        """
        Get an image with the skeleton and keypoints drawn.

        :return: an image with circles at keypoints and lines connecting keypoints,
                 representing the skeleton of the hand.
        """

        return self._draw_skeleton(keypoints=self.keypoints)

    # HEATMAPS

    def get_heatmap(self, key: int) -> np.ndarray:
        """
        Generate a heatmap for a specific keypoint.

        :param key: the index of the keypoint for which to generate the heatmap.
        :return: a 2D numpy array representing the heatmap of the specified keypoint.
        """

        # Creating empty heatmap
        new_size = DATA["new_size"]
        heatmap = np.zeros((new_size, new_size), dtype=np.float32)

        # Creating heatmap
        x0, y0 = self.keypoints[key]
        x = np.arange(0, new_size, 1, float)
        y = np.arange(0, new_size, 1, float)[:, np.newaxis]

        heatmap += np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2.0 * DATA["sigma_blur"] ** 2))

        return heatmap

    @property
    def heatmaps(self) -> np.ndarray:
        """
        Generate heatmaps for all keypoints.

        :return: A numpy array representing heatmaps for all keypoints.
        """

        return np.array([self.get_heatmap(key=i) for i in range(FREIHAND_INFO["n_keypoints"])])

    @property
    def _heatmaps_all(self) -> np.ndarray:
        """
        Generate a single heatmap with all keypoints.

        :return: single array representing an heatmap that combine the presence of all keypoints.
        """

        return np.sum(self.heatmaps, axis=0)

    # PREDICTION

    def predict_heatmap(self, model: nn.Module) -> np.ndarray[np.ndarray[float]]:
        """
        Generate predicted heatmaps for keypoints using a neural network model.

        :param model: A PyTorch neural network model for predicting heatmaps.

        :returns: array representing the predicted heatmaps for all keypoints.
                  The array shape is (num_keypoints, image_height, image_width).
        """

        # Convert the image to a numpy array and perform Z-score normalization
        img = self.image_arr_z
        img_transposed = np.transpose(a=img, axes=(2, 0, 1))
        img_tensor = torch.from_numpy(img_transposed).unsqueeze(0)

        # Get the predicted heatmaps from the model
        pred_heatmaps = model(img_tensor)[0].detach().numpy()

        return pred_heatmaps

    def predict_left_hand(self, model: nn.Module) -> bool:
        """
        Predicts if the hand is left using a neural network model.

        :param model: A PyTorch neural network classifier to discrimnate over left and right hand.

        :returns: True if the predicted hand is the left one, False otherwise.
        """

        # Convert the image to a numpy array and perform Z-score normalization
        img = self.image_arr_z
        img_transposed = np.transpose(a=img, axes=(2, 0, 1))
        X = torch.from_numpy(img_transposed)
        X = nn.functional.interpolate(X.unsqueeze(0), size=(227, 227), mode='bilinear', align_corners=False).squeeze()
        X = X.unsqueeze(0)

        predicted = model(X)

        is_left = torch.argmax(predicted, dim=1).item() == 0

        return is_left


    # PLOT

    def _plot(self, img_array: np.ndarray, title: str = ""):
        """
        Display image or heatmap.

        :param img_array: image or heatmap array to be displayed.
        :param title: title for the displayed image.
        """

        fig, ax = plt.subplots(1, 1, figsize=(3, 3))

        # Title
        ax.set_title(f"{self.idx} - {title}")
        ax.axis('off')

        # Heatmap
        if len(img_array.shape) == 2:
            plt.imshow(img_array, cmap='gray', interpolation='nearest')
        # Image
        else:
            plt.imshow(img_array)

        plt.show()

    def _plot2(self, img_arrays: Tuple[np.ndarray, np.ndarray],
               titles: Tuple[str, str], main_title: str):
        """
        Display two images side by side.

        :param img_arrays: tuple of two image or heatmap arrays.
        :param titles: titles for the two displayed images.
        :param main_title: main title for the entire plot.
        """

        # Unpacking
        img1, img2 = img_arrays
        title1, title2 = titles

        # Subplots
        fig, axes = plt.subplots(1, 2, figsize=(5, 3))

        for ax, img, title in zip([0, 1], [img1, img2], [title1, title2]):

            # Heatmap
            if len(img.shape) == 2:
                axes[ax].imshow(img, cmap='gray', interpolation='nearest')
            # Image
            else:
                axes[ax].imshow(img)

            axes[ax].axis('off')
            axes[ax].set_title(title)

        # Add a title for both images
        plt.suptitle(f"{self.idx} - {main_title}")

        plt.show()

    def plot_image(self):
        """
        Plot the original (raw) image.
        """

        self._plot(img_array=np.array(self.image), title="Raw")

    def plot_image_normalized(self):
        """
        Plot the normalized image.
        """

        self._plot(img_array=self.image_arr_z, title="Normalized")

    def plot_skeleton(self):
        """
        Plot the image with keypoints connected by a skeleton.
        """

        self._plot(img_array=np.array(self.skeleton), title="Skeleton")

    def plot_heatmaps(self):
        """
        Plot all keypoints heatmaps in a single image.
        """

        self._plot(img_array=self._heatmaps_all, title="Heatmaps")

    def plot_raw_skeleton(self):
        """
        Plot raw image and skeleton
        """

        self._plot2(
            img_arrays=(np.array(self.image), np.array(self.skeleton)),
            titles=("Raw image", "Skeleton"),
            main_title="Data sample"
        )

    def plot_network_input(self):
        """
        Plot the network input composed of the normalized image and heatmaps.
        """

        self._plot2(
            img_arrays=(self.image_arr_z, self._heatmaps_all),
            titles=("Normalized Image", "Heatmaps"),
            main_title="Network Input"
        )


class HandCollection:
    """
    This class represents a collection of Hand objects and provides methods to automate their generation.

    Attributes:
     - keypoints: list of keypoints for each hand image.
    """

    # CONSTRUCTOR

    def __init__(self):
        """
        Initialize a HandCollection object by reading and storing keypoints information.
        """

        # Store keypoints information
        self._keypoints: List[List[List[float]]] = read_json(get_2d_file())

    # REPRESENTATION

    def __str__(self) -> str:
        """
        Return a string representation of the HandCollection object.

        :return: string representation for the object.
        """

        return f"HandCollection"

    def __repr__(self) -> str:
        """
        Return a string representation of the HandCollection object.

        :return: string representation for the object.
        """

        return str(self)

    # ITEMS

    def __getitem__(self, idx: int) -> Hand:
        """
        Retrieve a specific Hand object from the collection by its index.

        :param idx: the index of the Hand object to retrieve.
        :return: hand object corresponding to the given index.
        """

        # Augmented labels are the same as raw ones
        raw_idx = idx % FREIHAND_INFO["raw"]

        # Create and return a Hand object
        return Hand(idx=idx, image=load_image(idx=idx), keypoints=self._keypoints[raw_idx])
