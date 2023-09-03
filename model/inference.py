"""
Inference over results
----------------------

This module provides a comprehensive toolkit for performing inference
 on hand pose estimation results using predicted heatmaps.
The toolkit includes classes and methods for loading models,processing hand images,obtaining predicted keypoints,
 calculating pixel errors, and visualizing results through heatmaps and skeleton images.

Classes:

- InferenceHand: A class for performing inference on hand pose estimation results.
- HandCollectionInference:  A class for managing a collection of hands and performing inference.
- ExternalHand: A class for performing inference on an hand image external to the Dataset.
"""

from typing import List, Tuple, Dict

import numpy as np
from PIL import Image

from io_ import load_image, log, load_external_image, load_model
from model.hand import Hand, HandCollection
from model.network import HandPoseEstimationUNet
from settings import FREIHAND_INFO, HANDPOSE_MODEL_CONFIG, DATA


class InferenceHand(Hand):
    """
    A class for performing inference on hand pose estimation results.

    This class extends the base `Hand` class to include methods for performing inference on predicted heatmaps and
    visualizing the results. It provides functionality for obtaining predicted keypoints, generating skeleton images,
    calculating pixel error, and plotting heatmaps and skeleton comparisons.

    Attribute:
    - pred_heatmaps: array of predicted heatmaps for keypoints.
    """

    # CONSTRUCTOR

    def __init__(self, idx: int | str, image: Image, keypoints: List,
                 pred_heatmaps: np.ndarray[np.ndarray[float]], mirrored: bool = False):
        """
        Initialize an InferenceHand instance.

        :param idx: dataset index of the hand image, or name of the file.
        :param image: PIL Image representing the hand image.
        :param keypoints: list of tuples containing ground truth keypoints.
        :param pred_heatmaps: array of predicted heatmaps for keypoints.
        :param mirrored: True if the hand is mirrored with respect to the original one, False otherwise.
        """

        super().__init__(idx=idx, image=image, keypoints=keypoints, mirrored=mirrored)
        self._pred_heatmaps = pred_heatmaps

    # REPRESENTATION

    def __str__(self) -> str:
        """
        Return a string representation of the Hand object.

        :returns: string representation of the object.
        """

        return f"InferenceHand[{self.idx} [{self.image_info}]"

    # MIRROR
    def mirror(self) -> Hand:

        mirrored_hand = super().mirror()

        mirrored_pred_heatmaps = np.array([np.flip(heatmap, axis=1) for heatmap in self.pred_heatmaps])

        return InferenceHand(
            idx=self.idx,
            image=mirrored_hand.image,
            keypoints=mirrored_hand.keypoints,
            pred_heatmaps=mirrored_pred_heatmaps,
            mirrored=not self.is_mirrored
        )

    # HEATMAPS

    def get_pred_heatmap(self, key: int) -> np.ndarray:
        """
        Returns the heatmap associated with a specific keypoint.

        :param key: Index of the desired keypoint.
        :return: heatmap array representing the confidence of the keypoint's location.
        """

        return self.pred_heatmaps[key]

    @property
    def pred_heatmaps(self) -> np.ndarray:
        """
        Returns an array containing heatmaps for all keypoints.

        :return: heatmap array for all keypoints.
        """

        return self._pred_heatmaps

    @property
    def _pred_heatmaps_all(self) -> np.ndarray:
        """
        Combine predicted heatmaps into a single heatmap array.

        :return: combined heatmap array.
        """

        return np.sum(self.pred_heatmaps, axis=0)

    # PREDICTIONS

    def _get_pred_keypoint(self, key: int) -> Tuple[float, float]:
        """
        Estimate the (x, y) coordinates of a predicted keypoint based on its heatmap.

        :param key: index of the keypoint.
        :return: tuple containing the predicted (x, y) coordinates.
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
        Returns a list of predicted keypoints.

        :return: list of tuples, each containing the (x, y) coordinates of a predicted keypoint.
        """

        return [self._get_pred_keypoint(key=i) for i in range(FREIHAND_INFO['n_keypoints'])]

    @property
    def pred_skeleton(self) -> Image:
        """
        Generates a skeleton image based on the predicted keypoints.

        :return: PIL Image representing the hand skeleton visualization.
        """

        return self._draw_skeleton(keypoints=self.pred_keypoints)

    # ERROR

    def _get_keypoint_error(self, key: int) -> float:
        """
        :param key: keypoint id
        :return: pixel error
        """

        x_true, y_true = self.keypoints[key]
        x_pred, y_pred = self.pred_keypoints[key]

        return np.sqrt((x_true - x_pred) ** 2 + (y_true - y_pred) ** 2)

    @property
    def mean_pixel_error(self) -> float:
        """
        Calculates the mean pixel error between predicted and ground truth keypoints.

        :return: mean pixel error value.
        """

        return float(np.mean(
            [self._get_keypoint_error(key=i) for i in range(FREIHAND_INFO["n_keypoints"])]
        ))

    # PLOT

    def plot_pred_heatmaps(self):
        """
        Plots all the predicted heatmaps in a single image for visual analysis.
        """

        self._plot(img_array=self._pred_heatmaps_all, title="Predicted heatmaps")

    def plot_pred_skeleton(self):
        """
        Plots the predicted skeleton image for visual comparison.
        """
        self._plot(img_array=np.array(self.pred_skeleton), title="Predicted skeleton")

    def plot_heatmaps_comparison(self):
        """
        Plots a comparison between the ground truth and predicted heatmaps.
        """

        self._plot2(img_arrays=(self._heatmaps_all, self._pred_heatmaps_all),
                    titles=("Ground truth", "Predicted"),
                    main_title="Model Heatmaps Prediction")

    def plot_skeletons_comparison(self):
        """
        Plots a comparison between the ground truth and predicted skeletons for visual assessment.
        """

        self._plot2(img_arrays=(np.array(self.skeleton), np.array(self.pred_skeleton)),
                    titles=("Ground truth", "Predicted"),
                    main_title="Model Heatmaps Prediction")


class HandCollectionInference(HandCollection):
    """
    A specialized class for managing a collection of hands and performing inference on hand pose estimation results.

    This class extends the `HandCollection` class and provides methods for loading a trained model,
    performing batch inference on multiple hand images, and retrieving `InferenceHand` instances for result interpretation.

    Attributes:
    - model: trained HandPoseEstimation network model.
    """

    # CONSTRUCTOR

    def __init__(self, config: Dict):
        """
        Initialize a HandCollectionInference instance.

        :param config: model configurations.
        """

        super().__init__()

        log(info="Loading the model")

        ann = HandPoseEstimationUNet(
            in_channel=config["in_channels"],
            out_channel=config["out_channels"]
        )

        self._model = load_model(model=ann, config=config)

    # REPRESENTATION

    def __str__(self) -> str:
        """
        Get string representation for HandPoseEstimationInference object

        :return: string representation for the object
        """

        return f"HandPoseEstimationInference"

    # ITEMS

    def __getitem__(self, idx: int) -> InferenceHand:
        """
        Retrieve an InferenceHand instance for result interpretation.

        :param idx: index for selecting a hand from the collection.
        :return: InferenceHand instance for the selected hand.
        """

        # Create collection
        collection = HandCollection()
        hand = collection[idx]

        # Evaluate prediction
        pred_heatmaps = hand.predict_heatmap(model=self._model)

        # Create InferenceHand

        raw_idx = idx % FREIHAND_INFO["raw"]

        return InferenceHand(
            idx=idx,
            image=load_image(idx=idx),
            keypoints=self._keypoints[raw_idx],
            pred_heatmaps=pred_heatmaps
        )


class ExternalHand:
    """
    A class for performing inference on an external hand image using a trained model.

    This class facilitates the inference process on an external hand image, generating predicted keypoints and visualizations
    based on a trained hand pose estimation model.

    Attributes:
    - file_name: name of file in external image directory.
    - hand: InferenceHand instance for the external hand image.
    """

    # CONSTRUCTOR

    def __init__(self, file_name: str, config: Dict):
        """
        Initialize an ExternalHand instance.

        :param file_name: file name or path of the external hand image.
        :param config: model configuration.
        """

        self._file_name: str = file_name
        self._hand: Infe = self._get_inference_hand(config=config)

    # REPRESENTATION

    def __str__(self):
        """
        Get string representation of ExternalHand object
        :return: string representation of the object
        """

        return f"ExternalHand[{self.idx} [{self._hand.image_info}]"

    def __repr__(self):
        """
        Get string representation of ExternalHand object
        :return: string representation of the object
        """

        return str(self)

    def _load_image(self) -> Image:
        """
        Load image from specific directory

        :return: loaded image
        """

        return load_external_image(file_name=self._file_name)

    def _get_inference_hand(self, config: Dict) -> InferenceHand:
        """
        Create an InferenceHand instance for the external hand image.

        :param config: model configuration.
        :return: InferenceHand instance for the external hand image.
        """

        # File path for the external image
        image = self._load_image()

        hand = Hand(
            idx=self._file_name,
            image=image,
            keypoints=[]
        )

        ann = HandPoseEstimationUNet(
            in_channel=config["in_channels"],
            out_channel=config["out_channels"]
        )        
        
        model = load_model(model=ann, config=config)
        pred_heatmaps = hand.predict_heatmap(model=model)

        return InferenceHand(
            idx=self._file_name,
            image=image,
            keypoints=[],
            pred_heatmaps=pred_heatmaps
        )

    def get_pred_heatmap(self, key: int) -> np.ndarray:
        """
        Get the heatmap for a specific keypoint.

        :param key: index of the desired keypoint.
        :return: heatmap array representing the confidence of the keypoint's location.
        """

        return self._hand.pred_heatmaps[key]

    @property
    def idx(self) -> str:
        """
        Get the index or identifier of the external hand.

        :return: Index or identifier of the external hand.
        """

        return self._file_name

    @property
    def pred_heatmaps(self) -> np.ndarray:
        """
        Returns an array containing heatmaps for all keypoints.

        :return: heatmap array for all keypoints.
        """

        return self._hand.pred_heatmaps

    @property
    def pred_keypoints(self) -> List[Tuple[float, float]]:
        """
        Returns a list of predicted keypoints.

        :return: list of tuples, each containing the (x, y) coordinates of a predicted keypoint.
        """

        return self._hand.pred_keypoints

    @property
    def pred_skeleton(self) -> Image:
        """
        Generates a skeleton image based on the predicted keypoints.

        :return: PIL Image representing the hand skeleton visualization.
        """

        return self._hand.pred_skeleton

    # PLOTS

    def plot_image(self):
        """
        Plot the original (raw) image.
        """

        self._hand.plot_image()

    def plot_pred_heatmaps(self):
        """
        Plots all the predicted heatmaps in a single image for visual analysis.
        """

        self._hand.plot_pred_heatmaps()

    def plot_pred_skeleton(self):
        """
        Plots a comparison between the ground truth and predicted heatmaps.
        """

        self._hand.plot_pred_skeleton()
