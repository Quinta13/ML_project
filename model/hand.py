"""
This file contains some classes to handle Hand and Keypoints
"""
from typing import List, Tuple

from PIL import Image, ImageDraw

from io_ import read_json, get_training_2d, read_image, log
from settings import ORIGINAL_SIZE, NEW_SIZE, FINGERS, COLORS, WIDTH, LINES, RAW, AUGMENTED


class Hand:

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


class HandCollection:

    def __init__(self):
        """

        :param only_raw: if to load only raw images
        :param n: number of images to load, if given only_raw is ignored
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
