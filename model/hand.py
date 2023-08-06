"""
This file contains some classes to handle Hand and Keypoints
"""
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw

from io_ import read_json, get_training_2d, read_image
from settings import ORIGINAL_SIZE, NEW_SIZE, FINGERS, COLORS, WIDTH, LINES, RAW, SIGMA_BLUR, NUM_KEYPOINTS, RADIUS, \
    POINT


class Hand:

    # DUNDERS

    def __init__(self, idx: int, image: Image, keypoints: List[List]):
        """

        :param idx: image dataset index
        :param image: image
        :param keypoints: keypoints
        """

        # Pad to 8 digits
        self._idx: str = str(idx).zfill(8)

        # Resize to NEW_SIZE
        self._image: Image = image.resize((NEW_SIZE, NEW_SIZE))
        self._keypoints: List[Tuple] = [
            (float(kp_x * NEW_SIZE / ORIGINAL_SIZE),
             float(kp_y * NEW_SIZE / ORIGINAL_SIZE))
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
        return int(self.idx) < RAW

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
        return self._idx

    @property
    def image(self) -> Image:
        """
        :return: image
        """
        return self._image

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

    @property
    def skeleton(self) -> Image:
        """
        :return: skeleton image
        """

        new_img = self.image.copy()
        draw = ImageDraw.Draw(new_img)

        color_point = tuple(int(POINT[i:i + 2], 16) for i in (0, 2, 4)) + (0,)  # from hex to rgb

        # Draw circles
        for keypoint in self.keypoints:

            x, y = keypoint

            # Calculate the bounding box of the circle
            x0 = x - RADIUS
            y0 = y - RADIUS
            x1 = x + RADIUS
            y1 = y + RADIUS

            # Draw the circle with the specified color and alpha
            draw.ellipse([(x0, y0), (x1, y1)], fill=color_point)

        # Draw lines
        for finger in FINGERS:

            # finger color
            color = COLORS[finger]
            color_rgb = tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))  # from hex to rgb

            # finger connections
            for line in LINES[finger]:
                p1, p2 = line
                draw.line([self.keypoints[p1], self.keypoints[p2]], fill=color_rgb, width=WIDTH)

        return new_img

    # HEATMAPS

    def get_heatmap(self, key: int) -> np.ndarray:
        """
        Returns the heatmap for given keypoint
        :param key: key index
        :return: keypoint heatmap array in scale [0, 1]
        """

        heatmap = np.zeros((NEW_SIZE, NEW_SIZE), dtype=np.float32)

        x0, y0 = self.keypoints[key]
        x = np.arange(0, NEW_SIZE, 1, float)
        y = np.arange(0, NEW_SIZE, 1, float)[:, np.newaxis]

        heatmap += np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2.0 * SIGMA_BLUR ** 2))

        return heatmap

    @property
    def heatmaps(self) -> np.ndarray:
        """
        Returns heatmaps of keypoints
        :return: all heatmaps array in scale [0, 1]
        """

        return np.array([self.get_heatmap(key=i) for i in range(NUM_KEYPOINTS)])

    @staticmethod
    def draw_heatmap(heatmap: np.ndarray) -> Image:
        """
        Draw heatmap given array of pixels in scale [0, 1]
        :param heatmap: heatmap array
        :return: heatmap image
        """

        scaled_image_array = (heatmap * 255).astype(np.uint8)

        # Convert the scaled array to a Pillow image with 'L' mode (grayscale)
        return Image.fromarray(scaled_image_array, mode='L')

    def heatmap_draw(self, key: int) -> Image:
        """
        Draws the heatmap for given keypoint
        :param key: key index
        :return: keypoint heatmap image
        """

        return self.draw_heatmap(
            heatmap=self.get_heatmap(key=key)
        )

    @property
    def heatmaps_draw(self) -> Image:
        """
        Draws all the heatmaps
        :return: keypoint heatmaps image
        """

        heatmap = np.sum(self.heatmaps, axis=0)

        return self.draw_heatmap(
            heatmap=heatmap
        )


class HandCollection:

    def __init__(self):
        """
        """

        self._keypoints: List[List[List[Tuple]]] = read_json(get_training_2d())

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

    def get_hand(self, idx: int) -> Hand:
        """
        Return hand with given index
        :param idx: index
        :return: hand
        """
        return Hand(idx=idx, image=read_image(idx=idx), keypoints=self._keypoints[idx % RAW])
