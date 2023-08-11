"""
This class provides some class to make inference over results
"""
from typing import List

import numpy as np
import torch
from PIL import Image

from io_ import read_image, read_json, get_2d_file, log
from model.dataset import FreiHANDDataset
from model.hand import Hand
from model.network import HandPoseEstimationUNet
from settings import FREIHAND_INFO, MODEL_CONFIG


class InferenceHand(Hand):

    def __init__(self, idx: int, image: Image,
                 keypoints: List[List[float]], pred_heatmaps: np.array):
        super().__init__(idx, image, keypoints)
        self._pred_heatmaps = pred_heatmaps

    def get_pred_heatmap(self, key: int) -> np.ndarray:
        """
        Returns the heatmap for given keypoint
        :param key: key index
        :return: keypoint heatmap array in scale [0, 1]
        """

        return self.heatmaps[key]

    @property
    def pred_heatmaps(self) -> np.ndarray:
        """
        Returns heatmaps of keypoints
        :return: all heatmaps array in scale [0, 1]
        """

        return self._pred_heatmaps

    @property
    def pred_heatmaps_all(self) -> np.ndarray:
        """
        Returns heatmaps in a single array
        :return: heatmaps array
        """

        return np.sum(self.pred_heatmaps, axis=0)

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

        self._plot(img_array=self.heatmaps_all)


class HandPoseEstimationInference:

    def __init__(self, model_fp: str):
        """

        :param model_fp: file path to trained HandPoseEstimation network
        """

        log(info="Loading the model")

        # define the model
        model = HandPoseEstimationUNet(
            in_channel=MODEL_CONFIG["in_channels"],
            out_channel=MODEL_CONFIG["out_channels"]
        )

        # load the model from memory
        model.load_state_dict(
            state_dict=torch.load(
                f=model_fp,
                map_location=MODEL_CONFIG["device"]
            )
        )
        model.eval()

        self._model = model
        self._dataset = FreiHANDDataset(set_type='test')
        self._keypoints: List[List[List[float]]] = read_json(get_2d_file())

    def __str__(self) -> str:
        """
        :return: string representation for the object
        """
        return f"HandPoseEstimationInference [{len(self)} items]"

    def __repr__(self) -> str:
        """
        :return: string representation for the object
        """
        return str(self)

    def __len__(self) -> int:
        """
        :return: dataset length
        """
        return len(self._dataset)

    def __getitem__(self, idx: int) -> InferenceHand:
        """
        Return hand for result interpretation
        :param idx: index for the test set
        :return: hand for inference
        """

        item = self._dataset[idx]

        image_name = item["image_name"]

        actual_idx = int(image_name[:-(len(FREIHAND_INFO["ext"]) + 1)])

        # augmented labels are the same as raw ones
        raw_idx = actual_idx % FREIHAND_INFO["raw"]

        # computing predicted heatmap
        image = item["image"].unsqueeze(0)
        pred_heatmaps = self._model(image).detach().numpy()

        return InferenceHand(
            idx=actual_idx,
            image=read_image(idx=actual_idx),
            keypoints=self._keypoints[raw_idx],
            pred_heatmaps=pred_heatmaps
        )
