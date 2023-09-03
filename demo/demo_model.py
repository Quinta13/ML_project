from os import path
from typing import Dict

from PIL import Image

from model.inference import ExternalHand
from model.left_hand import ExternalLeftHand

DEMO_DIR = str(path.abspath(path.join(__file__, "../")))


class DemoHand(ExternalHand):
    """
    A class for performing inference on a demo hand image using a trained model.

    This class facilitates the inference process on a demo hand image,
    generating predicted keypoints and visualizations based on a trained hand pose estimation model.

    Attributes:
    - file_name: name of file in demo image directory.
    - hand: InferenceHand instance for the external hand image.
    """

    # CONSTRUCTOR

    def __init__(self, file_name: str, config: Dict):
        """
        Initialize an ExternalHand instance.

        :param file_name: file name or path of the external hand image.
        :param config: model configuration.
        """

        super().__init__(file_name, config)

    # REPRESENTATION

    def __str__(self):
        """
        Get string representation of ExternalHand object
        :return: string representation of the object
        """

        return f"DemoHand[{self.idx} [{self._hand.image_info}]"

    def _load_image(self) -> Image:
        """
        Load image from specific directory

        :return: loaded image
        """

        # File path for the external image
        in_path = path.join(DEMO_DIR, self._file_name)
        return Image.open(fp=in_path)


class DemoLeftHand(ExternalLeftHand):
    """
    A class for performing inference on a demo left hand image using a trained model.

    This class facilitates the inference process on a demo left hand image,
    generating predicted keypoints and visualizations based on a trained hand pose estimation model.

    Attributes:
    - file_name: name of file in demo image directory.
    - hand: InferenceHand instance for the external hand image.
    """

    # CONSTRUCTOR

    def __init__(self, file_name: str, estimator_config: Dict, classifier_config: Dict):
        """
        Initialize an ExternalLeftHand instance.

        :param file_name: file name or path of the external hand image.
        :param classifier_config: model configurations for left vs. right hand classifier.
        :param estimator_config: model configurations for 2-hand pose estimation.
        """

        super().__init__(file_name=file_name, estimator_config=estimator_config, classifier_config=classifier_config)

    # REPRESENTATION

    def __str__(self):
        """
        Get string representation of ExternalHand object
        :return: string representation of the object
        """

        return f"DemoLeftHand[{self.idx} [{self._hand.image_info}]"

    def _load_image(self) -> Image:
        """
        Load image from specific directory

        :return: loaded image
        """

        # File path for the external image
        in_path = path.join(DEMO_DIR, self._file_name)
        return Image.open(fp=in_path)

