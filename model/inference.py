"""
This class provides some class to make inference over results
"""
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt

from io_ import read_image, read_json, get_2d_file, log, read_external_image, load_model
from model.dataset import FreiHANDDataset
from model.hand import Hand, HandCollection
from model.network import HandPoseEstimationUNet
from settings import FREIHAND_INFO, MODEL_CONFIG
from utlis import from_image_to_heatmap


class InferenceHand(Hand):

    def __init__(self, idx: int, image: Image,
                 keypoints: List, pred_heatmaps: np.ndarray[np.ndarray[float]]):
        super().__init__(idx, image, keypoints)
        self._pred_heatmaps = pred_heatmaps

    def get_pred_heatmap(self, key: int) -> np.ndarray:
        """
        Returns the heatmap for given keypoint
        :param key: key index
        :return: keypoint heatmap array in scale [0, 1]
        """

        return self.pred_heatmaps[key]

    @property
    def pred_heatmaps(self) -> np.ndarray:
        """
        Returns heatmaps of keypoints
        :return: all heatmaps array in scale [0, 1]
        """

        return self._pred_heatmaps

    @property
    def _pred_heatmaps_all(self) -> np.ndarray:
        """
        Returns heatmaps in a single array
        :return: heatmaps array
        """

        return np.sum(self.pred_heatmaps, axis=0)

    def _get_pred_keypoint(self, key: int) -> Tuple[float, float]:
        """
        Estimate keypoint from the heatmap
        :param key: keypoint id
        :return: predicted keypoint
        """

        heatmap = self.get_pred_heatmap(key=key)

        heatmap_nrm: np.ndarray[np.ndarray[float]] = heatmap / np.sum(heatmap)

        heatmap_rows: np.ndarray[float] = np.sum(heatmap_nrm, axis=0)
        heatmap_cols: np.ndarray[float] = np.sum(heatmap_nrm, axis=1)

        size = len(heatmap)

        kx: float = float(np.dot(np.arange(size), heatmap_rows))
        ky: float = float(np.dot(np.arange(size), heatmap_cols))

        return kx, ky

    @property
    def pred_keypoints(self) -> List[Tuple[float, float]]:
        """
        :return: predicted keypoints
        """
        return [self._get_pred_keypoint(key=i) for i in range(FREIHAND_INFO['n_keypoints'])]

    @property
    def pred_skeleton(self) -> Image:
        """
        :return: skeleton image
        """

        return self._draw_skeleton(keypoints=self.pred_keypoints)

    def _keypoint_distance(self, key: int) -> float:
        """
        :param key: keypoint id
        :return: pixel error
        """

        x_true, y_true = self.keypoints[key]
        x_pred, y_pred = self.pred_keypoints[key]

        return np.sqrt((x_true - x_pred) ** 2 + (y_true - y_pred) ** 2)

    @property
    def mean_pixel_distance(self):
        """
        :return: mean pixel distance
        """

        return np.mean(
            [self._keypoint_distance(key=i) for i in range(FREIHAND_INFO["n_keypoints"])]
        )

    def plot_pred_heatmap(self, key: int):
        """
        Plots the heatmap for given keypoint
        :param key: keypoint index
        """

        self._plot(img_array=self.get_pred_heatmap(key=key))

    def plot_pred_heatmaps(self):
        """
        Plots all the heatmaps in a single image
        """

        self._plot(img_array=self._pred_heatmaps_all)

    def plot_pred_skeleton(self):
        self._plot(img_array=np.array(self.pred_skeleton))

    def plot_heatmaps_comparison(self):
        """
        Plots heatmaps ground truth and predicted
        """

        # Subplots
        fig, axes = plt.subplots(1, 2, figsize=(10, 10))

        # Plot original image
        axes[0].imshow(self._heatmaps_all, cmap='gray')
        axes[0].set_title('Ground_truth')
        axes[0].axis('off')

        # Plot original image
        axes[1].imshow(self._pred_heatmaps_all, cmap='gray')
        axes[1].set_title('Predicted')
        axes[1].axis('off')

    def plot_skeletons_comparison(self):
        """
        Plots heatmaps ground truth and predicted
        """

        # Subplots
        fig, axes = plt.subplots(1, 2, figsize=(10, 10))

        # Plot original image
        axes[0].imshow(self.skeleton)
        axes[0].set_title('Ground_truth')
        axes[0].axis('off')

        # Plot original image
        axes[1].imshow(self.pred_skeleton)
        axes[1].set_title('Predicted')
        axes[1].axis('off')


class HandCollectionInference(HandCollection):
    """
    This class ... TODO
    """

    def __init__(self, model_fp: str):
        """

        :param model_fp: file path to trained HandPoseEstimation network
        """

        super().__init__()

        log(info="Loading the model")

        self._model = load_model(path_=model_fp)

    def __str__(self) -> str:
        """
        :return: string representation for the object
        """
        return f"HandPoseEstimationInference"

    def __repr__(self) -> str:
        """
        :return: string representation for the object
        """
        return str(self)

    def __getitem__(self, idx: int) -> InferenceHand:
        """
        Return hand for result interpretation
        :param idx: index for the test set
        :return: hand for inference
        """

        collection = HandCollection()
        hand = collection[idx]

        # Evaluating prediction

        pred_heatmaps = from_image_to_heatmap(hand, self._model)

        raw_idx = idx % FREIHAND_INFO["raw"]

        return InferenceHand(
            idx=idx,
            image=read_image(idx=idx),
            keypoints=self._keypoints[raw_idx],
            pred_heatmaps=pred_heatmaps
        )


class ExternalHand:
    """ TODO """

    def __init__(self, file_name: str, model_fp: str):
        """
        TODO
        """

        self._file_name = file_name

        self._hand = self._get_inference_hand(model_fp=model_fp)

    def _get_inference_hand(self, model_fp) -> InferenceHand:

        image = read_external_image(file_name=self._file_name)

        hand = Hand(
            idx=0,
            image=image,
            keypoints=[]
        )

        model = load_model(model_fp)
        pred_heatmaps = from_image_to_heatmap(hand, model)

        return InferenceHand(
            idx=0,
            image=image,
            keypoints=[],
            pred_heatmaps=pred_heatmaps
        )

    def get_pred_heatmap(self, key: int) -> np.ndarray:
        """
        Returns the heatmap for given keypoint
        :param key: key index
        :return: keypoint heatmap array in scale [0, 1]
        """

        return self._hand.pred_heatmaps[key]

    @property
    def pred_heatmaps(self) -> np.ndarray:
        """
        Returns heatmaps of keypoints
        :return: all heatmaps array in scale [0, 1]
        """

        return self._hand.pred_heatmaps

    @property
    def pred_keypoints(self) -> List[Tuple[float, float]]:
        """
        :return: predicted keypoints
        """
        return self._hand.pred_keypoints

    @property
    def pred_skeleton(self) -> Image:
        """
        :return: skeleton image
        """

        return self._hand.pred_skeleton

    def plot_image(self):
        self._hand.plot_image()

    def plot_pred_heatmap(self, key: int):
        """
        Plots the heatmap for given keypoint
        :param key: keypoint index
        """

        self._hand.plot_pred_heatmap(key=key)

    def plot_pred_heatmaps(self):
        """
        Plots all the heatmaps in a single image
        """

        self._hand.plot_pred_heatmaps()

    def plot_pred_skeleton(self):
        self._hand.plot_pred_skeleton()
