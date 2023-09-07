import os
from os import path
import time
from typing import List, Dict

import cv2
import numpy as np
from PIL import Image
from tabulate import tabulate


""" PATHS TO RUN WITH TERMINAL """
import sys
sys.stderr = open(os.devnull, 'w')
demo_dir = path.abspath(path.dirname(path.join(__file__, "../")))
root_dir = path.abspath(path.dirname(path.join(__file__, "../", "../")))
sys.path.remove(demo_dir)
sys.path.append(root_dir)

from io_ import load_model
from model.network import HandPoseEstimationUNet
from model.left_hand import AlexNet, ExternalLeftHand
from model.hand import HandCollection
from model.inference import InferenceHand, ExternalHand
from settings import HANDPOSE_MODEL_CONFIG, LEFT_RIGHT_MODEL_CONFIG, DATA


# COLORS
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[96m"
RESET = "\033[0m"

# GLOBALS SETTING
DEMO_DIR = str(path.abspath(path.join(__file__, "../")))
CONTINUE_KEY = 32  # space-bar
IN = path.join(DEMO_DIR, "in")
SCREENSHOT_FP = path.join(DEMO_DIR, IN, "img.jpg")
OUT_SIZE = (440, 440)
CAMERA = 2

# MODELS
CLASSIFIER = load_model(model=AlexNet(num_classes=2), config=LEFT_RIGHT_MODEL_CONFIG)
ESTIMATOR = load_model(
    model=HandPoseEstimationUNet(in_channel=HANDPOSE_MODEL_CONFIG["in_channels"],
                                 out_channel=HANDPOSE_MODEL_CONFIG["out_channels"]),
    config=HANDPOSE_MODEL_CONFIG
)
COLLECTION = HandCollection()


# CLASSES
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

        super().__init__(file_name=file_name, config=config)

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


# PREDICTIONS
def show_predictions(img1: Image, img2: Image):

    img1 = img1.resize(OUT_SIZE, Image.ANTIALIAS)
    opencv_img1 = cv2.cvtColor(np.array(img1), cv2.COLOR_BGR2RGB)

    img2 = img2.resize(OUT_SIZE, Image.ANTIALIAS)
    opencv_img2 = cv2.cvtColor(np.array(img2), cv2.COLOR_BGR2RGB)

    # Display the image using OpenCV
    cv2.imshow('HandPoseEstimation', opencv_img1)
    cv2.imshow('HandPoseEstimation 2.0', opencv_img2)

    # Move
    cv2.moveWindow('HandPoseEstimation', 0, 0)
    cv2.moveWindow('HandPoseEstimation 2.0', 500, 500)

    # Display until continue key is pressed
    while True:
        k = cv2.waitKey(1)
        if k == CONTINUE_KEY:
            cv2.destroyAllWindows()
            break


def show_predictions4(img1: Image, img2: Image, img3: Image, img4: Image):

    img1 = img1.resize(OUT_SIZE, Image.ANTIALIAS)
    opencv_img1 = cv2.cvtColor(np.array(img1), cv2.COLOR_BGR2RGB)

    img2 = img2.resize(OUT_SIZE, Image.ANTIALIAS)
    opencv_img2 = cv2.cvtColor(np.array(img2), cv2.COLOR_BGR2RGB)

    img3 = img3.resize(OUT_SIZE, Image.ANTIALIAS)
    opencv_img3 = cv2.cvtColor(np.array(img3), cv2.COLOR_BGR2RGB)

    img4 = img4.resize(OUT_SIZE, Image.ANTIALIAS)
    opencv_img4 = cv2.cvtColor(np.array(img4), cv2.COLOR_BGR2RGB)

    # Display the image using OpenCV
    cv2.imshow('Raw', opencv_img1)
    cv2.imshow('Augmented1', opencv_img2)
    cv2.imshow('Augmented2', opencv_img3)
    cv2.imshow('Augmented3', opencv_img4)

    # Move
    cv2.moveWindow('Raw', 0, 0)
    cv2.moveWindow('Augmented1', 500, 0)
    cv2.moveWindow('Augmented2', 0, 500)
    cv2.moveWindow('Augmented3', 500, 500)

    # Display until continue key is pressed
    while True:
        k = cv2.waitKey(1)
        if k == CONTINUE_KEY:
            cv2.destroyAllWindows()
            break

# DEMO IMAGES

def _demo_images(img_fp):

    # Load hand
    hand1 = DemoHand(
        file_name=img_fp,
        config=HANDPOSE_MODEL_CONFIG
    )

    # Load left-hand
    hand2 = DemoLeftHand(
        file_name=img_fp,
        estimator_config=HANDPOSE_MODEL_CONFIG,
        classifier_config=LEFT_RIGHT_MODEL_CONFIG
    )

    show_predictions(
        img1=hand1.pred_skeleton,
        img2=hand2.pred_skeleton
    )


def demo_images():

    # Demo images in IN directory
    images: List[str] = os.listdir(IN)
    images.sort()
    images_dict: Dict[int, str] = {i + 1: img for i, img in enumerate(images)}

    print()
    print(f"{YELLOW}---- Demo Images ----{RESET}")

    menu = [[f"{BLUE}[{i}]{RESET}", img] for i, img in images_dict.items()]
    print(tabulate(menu, tablefmt="fancy_grid"))

    print()

    i = input(f"{GREEN}• Enter image option: {RESET}")

    if not i.isdigit():
        print()
        input(f"  {RED}ERR{RESET}: Please select a number option...")
        return

    i = int(i)

    if i < 1 or len(images_dict) < i:
        print()
        input(f"  {RED}ERR{RESET}: Please select a valid index in specified image...")
        return
    
    img_name = images_dict[i]
    img_fp = path.join(IN, img_name)

    _demo_images(img_fp=img_fp)


# FREIHAND

def freihand():

    print()
    print(f"{YELLOW}---- FreiHAND Dataset ----{RESET}")

    print()
    idx = input(f"{GREEN}• Enter image index{YELLOW} [0, 130.239]{RESET}: ")

    if not idx.isdigit():
        print()
        input(f"  {RED}ERR{RESET}: Please select a number as image index...")
        return

    idx = int(idx)

    if idx < 0 or 130239 < idx:
        print()
        input(f"  {RED}ERR{RESET}: Please select an image index in specified range...")
        return

    print()
    mirror_ch = input(f"{GREEN}• Mirror to simulate left hand?{RESET} {YELLOW}[y/n]{RESET}: ")

    if mirror_ch == "y":
        mirror = True
    elif mirror_ch == "n":
        mirror = False
    else:
        print()
        input(f"  {RED}ERR{RESET}: Choose one between \"y\" and \"n\"...")
        return

    hand1 = COLLECTION[idx]
    hand2 = COLLECTION[idx]

    if mirror:
        hand1 = hand1.mirror()
        hand2 = hand2.mirror()

    left = hand2.predict_left_hand(model=CLASSIFIER)

    if left:
        hand2 = hand2.mirror()

    pred_hand1 = InferenceHand(
        idx=hand1.idx,
        image=hand1.image,
        keypoints=hand1.keypoints,
        pred_heatmaps=hand1.predict_heatmap(model=ESTIMATOR)
    )

    pred_hand2 = InferenceHand(
        idx=hand2.idx,
        image=hand2.image,
        keypoints=hand2.keypoints,
        pred_heatmaps=hand2.predict_heatmap(model=ESTIMATOR)
    )

    if left:
        pred_hand2 = pred_hand2.mirror()

    show_predictions(img1=pred_hand1.pred_skeleton,
                     img2=pred_hand2.pred_skeleton)
    

# AUGMENTATION

def augmentation():

    print()
    print(f"{YELLOW}---- Data Augmentation ----{RESET}")
    
    print()
    idx = input(f"{GREEN}• Enter image index{YELLOW} [0, 32.559]{RESET}: ")

    if not idx.isdigit():
        print()
        input(f"  {RED}ERR{RESET}: Please select a number as image index...")
        return

    idx = int(idx)

    if idx < 0 or 32559 < idx:
        print()
        input(f"  {RED}ERR{RESET}: Please select an image index in specified range...")
        return

    hands = [COLLECTION[idx + i * 32560] for i in range(4)] 

    inference_hands = [
        InferenceHand(
            idx=hand.idx,
            image=hand.image,
            keypoints=hand.keypoints,
            pred_heatmaps=hand.predict_heatmap(model=ESTIMATOR)
        )
        for hand in hands
    ]

    img1, img2, img3, img4 = [hand.pred_skeleton for hand in inference_hands]

    show_predictions4(img1=img1, img2=img2, img3=img3, img4=img4)

                      
# SCREENSHOT

def img_transformation():
    """
    Transform screenshot image in a format that can be processed by the ANN architecutre
    """

    # Read image
    img = cv2.imread(SCREENSHOT_FP)
    H, W, _ = img.shape

    # Size of the center square
    size = min(H, W)

    # Starting point to crop the center square
    x_start = (W - size) // 2
    y_start = (H - size) // 2

    # Crop the center square
    center_square = img[y_start:y_start + size, x_start:x_start + size]

    # Resize the center square to WxH
    resized_image = cv2.resize(center_square, (W, H))

    # Resize to model input
    new_size = DATA["new_size"]
    resized_image = cv2.resize(resized_image, (new_size, new_size))

    # Save
    cv2.imwrite(SCREENSHOT_FP, resized_image)


def screenshot():
    """
    Uses camera to take the screenshot
    """

    # Initialize the USB webcam
    cam = cv2.VideoCapture(CAMERA)
    cv2.namedWindow('2-Hand Pose Estimation - Demo')
    
    while True:

        # Initializing the frame and ret
        ret, frame = cam.read()

        # If statement
        if not ret:
            print('Failed to grab frame')
            break

        # The frame will show with the title of the app
        cv2.imshow('2-Hand Pose Estimation - Screenshot', frame)

        # Wait for key to be pressed
        k = cv2.waitKey(1)

        # If the continue key is pressed, screenshots will be taken
        if k == CONTINUE_KEY:

            print(f'Screenshot taken and saved as {SCREENSHOT_FP}')

            # Save screenshot and transform
            cv2.imwrite(SCREENSHOT_FP, frame)
            img_transformation()

            # Close camera
            cam.release()
            cv2.destroyAllWindows()

            # Show outputs
            _demo_images(SCREENSHOT_FP)

            # Delete image
            os.remove(SCREENSHOT_FP)

            break

# MENU


if __name__ == "__main__":

    # Base options
    menu_options = ["FreiHandCollection", "Augmentation", "Directory", "Screenshot"]

    while True:

        os.system("clear")

        print(f"{YELLOW} -- 2D Hand Pose Estimation -- {RESET}")

        menu = [[f"{BLUE}[{i+1}]{RESET}", option] for i, option in enumerate(menu_options)]
        menu += [[f"{BLUE}[0]{RESET}", "Exit"]]
        print(tabulate(menu, tablefmt="fancy_grid"))

        print()
        choice = input(f"{GREEN}• Enter option: {RESET}")

        if choice.isdigit():
            choice = int(choice)
              
            if choice == 0:
                print()
                print("Thank you! ")
                time.sleep(2)
                os.system('clear')
                
                break

            elif choice == 1:
                freihand()

            elif choice == 2:
                augmentation()

            elif choice == 3:
                demo_images()

            elif choice == 4:
                screenshot()
                
            else:
                print()
                input(f"  {RED}ERR{RESET}: Please choose a number between {0} and {len(menu_options)}...")
        else:
            print()
            input(f"  {RED}ERR{RESET}: Please insert a number to select an option...")

    os.system("clear")
