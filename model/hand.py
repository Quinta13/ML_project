"""
This file contains some classes to handle Hand image, transformations, keypoints and heatmaps
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Union

import numpy as np
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

from io_ import read_json, read_image, get_2d_file, read_means_stds
from settings import DATA, FREIHAND_INFO
from utlis import pad_idx

# CONNECTIONS
KEYPOINTS_CONNECTIONS: Dict[str, List[Tuple[int, int]]] = {
    "thumb": [(0, 1), (1, 2), (2, 3), (3, 4)],
    "index": [(0, 5), (5, 6), (6, 7), (7, 8)],
    "middle": [(0, 9), (9, 10), (10, 11), (11, 12)],
    "ring": [(0, 13), (13, 14), (14, 15), (15, 16)],
    "little": [(0, 17), (17, 18), (18, 19), (19, 20)],
}

# DRAW STYLE

STYLE: Dict[str, int | float | str] = {
    "point_color": "383838",
    "point_radius": 1.5,
    "line_width": 1
}

COLORS: Dict[str, str] = {
    "thumb": "008000",
    "index": "00FFFF",
    "middle": "0000FF",
    "ring": "FF00FF",
    "little": "FF0000"
}


class Hand:

    """
    This class read information about an Hand in terms of image and keypoints, moreover it can
        - apply Z-transformation to images
        - generate heatmaps for the keypoints
    """


    # DUNDERS

    def __init__(self, idx: int, image: Image, keypoints: List):
        """
        It read the image and get the keypoints,
            it rescale them to NEW_SIZE from settings
        :param idx: image dataset index
        :param image: image
        :param keypoints: keypoints
        """

        # Pad to 8 digits
        self._idx: str = pad_idx(idx=idx)

        # Resize to NEW_SIZE
        new_size = DATA["new_size"]
        original_size = FREIHAND_INFO["size"]

        self._image: Image = image.resize((new_size, new_size))
        self._keypoints: List[Tuple[float, float]] = [
            (float(kp_x * new_size / original_size),
             float(kp_y * new_size / original_size))
            for kp_x, kp_y in keypoints
        ]

    def __str__(self) -> str:
        """
        :return: string representation for the object
        """
        return f"Image {self.idx} [{self.image_info}]"

    def __repr__(self) -> str:
        """
        :return: string representation for the object
        """
        return str(self)

    # PROPERTIES

    @property
    def is_raw(self) -> bool:
        """
        :return: if image is raw
        """
        return int(self.idx[:-4]) < FREIHAND_INFO["raw"]

    @property
    def is_augmented(self) -> bool:
        """
        :return: if image is augmented
        """
        return not self.is_raw

    @property
    def idx(self) -> str:
        """
        :return: image dataset index
        """
        return f"{self._idx}.{FREIHAND_INFO['ext']}"

    @property
    def image(self) -> Image:
        """
        :return: image
        """
        return self._image

    @property
    def image_arr(self) -> np.ndarray:
        """
        :return: image as array
        """
        return np.array(self._image).astype(dtype=np.float32)

    @property
    def image_arr_mm(self) -> np.ndarray:
        """
        :return: image min-max scaled
        """
        return self.image_arr / 255

    @property
    def image_arr_z(self) -> np.ndarray:
        """
        :return: Z-transformation to image
        """

        means, stds = read_means_stds()

        img = self.image_arr_mm

        r_channel, g_channel, b_channel = img[:, :, 0], img[:, :, 1], img[:, :, 2]

        # Z-transform each channel with means and stds
        r_channel = (r_channel - means[0]) / stds[0]
        g_channel = (g_channel - means[1]) / stds[1]
        b_channel = (b_channel - means[2]) / stds[2]

        # Merge back into an image
        img_z = np.stack((r_channel, g_channel, b_channel), axis=-1)

        return img_z

    @property
    def image_info(self) -> str:
        """
        :return: information about the image
        """
        return f"Size: {self.image.size}, Mode: {self.image.mode}"

    @property
    def keypoints(self) -> List[Tuple]:
        """
        :return: keypoints
        """
        return self._keypoints

    def _draw_skeleton(self, keypoints: List[Tuple[float, float]]) -> Image:
        """
        Draws given keypoints to the skeleton
        :param keypoints: list of keypoints
        :return: image with skeleton
        """

        new_img = self.image.copy()
        draw = ImageDraw.Draw(new_img)

        color_point = tuple(int(STYLE["point_color"][i:i + 2], 16) for i in (0, 2, 4)) + (0,)  # from hex to rgb

        # Draw circles
        for keypoint in keypoints:

            x, y = keypoint

            # Calculate the bounding box of the circle
            radius = STYLE["point_radius"]
            x0 = x - radius
            y0 = y - radius
            x1 = x + radius
            y1 = y + radius

            # Draw the circle with the specified color and alpha
            draw.ellipse([(x0, y0), (x1, y1)], fill=color_point)

        # Draw lines
        for finger_key, connection in KEYPOINTS_CONNECTIONS.items():

            # finger color
            color = COLORS[finger_key]
            color_rgb = tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))  # from hex to rgb

            # finger connections
            for line in connection:
                p1, p2 = line
                draw.line([keypoints[p1], keypoints[p2]], fill=color_rgb, width=STYLE["line_width"])

        return new_img

    @property
    def skeleton(self) -> Image:
        """
        :return: skeleton image
        """

        return self._draw_skeleton(keypoints=self.keypoints)

    # HEATMAPS

    def get_heatmap(self, key: int) -> np.ndarray:
        """
        Returns the heatmap for given keypoint
        :param key: key index
        :return: keypoint heatmap array in scale [0, 1]
        """

        new_size = DATA["new_size"]

        heatmap = np.zeros((new_size, new_size), dtype=np.float32)

        x0, y0 = self.keypoints[key]
        x = np.arange(0, new_size, 1, float)
        y = np.arange(0, new_size, 1, float)[:, np.newaxis]

        heatmap += np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2.0 * DATA["sigma_blur"] ** 2))

        return heatmap

    @property
    def heatmaps(self) -> np.ndarray:
        """
        Returns heatmaps of keypoints
        :return: all heatmaps array in scale [0, 1]
        """

        return np.array([self.get_heatmap(key=i) for i in range(FREIHAND_INFO["n_keypoints"])])

    @property
    def _heatmaps_all(self) -> np.ndarray:
        """
        Returns heatmaps in a single array
        :return: heatmaps array
        """

        return np.sum(self.heatmaps, axis=0)

    # PLOT

    @staticmethod
    def _plot(img_array: np.ndarray):
        """
        Draw image
        :param img_array: heatmap array
        """

        if len(img_array.shape) == 2:  # grayscale
            plt.imshow(img_array, cmap='gray', interpolation='nearest')
        else:  # rgb
            plt.imshow(img_array)
        plt.axis('off')

    def plot_image(self):
        """
        Plots original image
        """
        self._plot(img_array=np.array(self.image))

    def plot_image_normalized(self):
        """
        Plots original image
        """
        self._plot(img_array=self.image_arr_z)

    def plot_skeleton(self):
        """
        Plots skeleton
        """
        self._plot(img_array=np.array(self.skeleton))

    def plot_heatmap(self, key: int):
        """
        Plots the heatmap for given keypoint
        :param key: keypoint index
        """

        self._plot(img_array=self.get_heatmap(key=key))

    def plot_heatmaps(self):
        """
        Plots all the heatmaps in a single image
        """

        self._plot(img_array=self._heatmaps_all)

    def plot_network_input(self):
        """
        Plots input for the network
        """

        means, stds = read_means_stds()

        # Subplots
        fig, axes = plt.subplots(1, 2, figsize=(10, 10))

        # Plot original image
        axes[0].imshow(self.image_arr_z)
        axes[0].set_title('Feature vector')
        axes[0].axis('off')

        # Plot original image
        axes[1].imshow(self._heatmaps_all, cmap='gray')
        axes[1].set_title('Heatmaps')
        axes[1].axis('off')

    def plot_preparation(self):
        """
        Draws
        :param means: mean for every channel
        :param stds: standard deviation for every channel
        """

        # Subplots
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        # Plot original image
        axes[0][0].imshow(self.image)
        axes[0][0].set_title('Original Image')
        axes[0][0].axis('off')

        # Plot original image
        axes[0][1].imshow(self.skeleton)
        axes[0][1].set_title('Skeleton')
        axes[0][1].axis('off')

        # Plot Z-transformation
        axes[1][0].imshow(self.image_arr_z)
        axes[1][0].set_title('Feature vector')
        axes[1][0].axis('off')

        # Plot heatmaps
        heatmap = np.sum(self.heatmaps, axis=0)
        axes[1][1].imshow(heatmap, cmap='gray')
        axes[1][1].set_title('Labels - Heatmaps')
        axes[1][1].axis('off')


class HandCollection:

    """
    This class automatizes Hand class generation:
     - it keeps in memory keypoint file
     - it automatizes raw-augmented labeling handling
    """

    def __init__(self):
        """
        It basically reads keypoints file
        """

        # we keep the file in memory and we read once
        self._keypoints: List[List[List[float]]] = read_json(get_2d_file())

    def __str__(self) -> str:
        """
        :return: string representation for the object
        """
        return f"HandCollection"

    def __repr__(self) -> str:
        """
        :return: string representation for the object
        """
        return str(self)

    def __getitem__(self, idx: int) -> Hand:
        """
        Return hand with given index
        :param idx: index
        :return: hand
        """

        # augmented labels are the same as raw ones
        raw_idx = idx % FREIHAND_INFO["raw"]

        return Hand(idx=idx, image=read_image(idx=idx), keypoints=self._keypoints[raw_idx])

