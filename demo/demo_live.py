import os
from os import path

import cv2
import numpy as np
from PIL import Image

from demo.demo_model import DEMO_DIR, DemoHand, DemoLeftHand
from settings import HANDPOSE_MODEL_CONFIG, LEFT_RIGHT_MODEL_CONFIG, DATA

CONTINUE_KEY = 32  # space-bar
EXIT_KEY = 27  # esc
OUT_SIZE = (480, 480)
CAMERA = 0
IMAGE_FP = path.join(DEMO_DIR, "img.jpg")


def img_transformation():
    """
    Transform screenshot image in a format that can be processed by the ANN architecutre
    """

    # Read image
    img = cv2.imread(IMAGE_FP)
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
    cv2.imwrite(IMAGE_FP, resized_image)


def take_screenshot():
    """
    Uses camera to take the screenshot
    """

    # Initialize the USB webcam
    cam = cv2.VideoCapture(CAMERA)
    cv2.namedWindow('2-Hand Pose Estimation - Demo')

    # Initializing the frame and ret
    ret, frame = cam.read()

    # If statement
    if not ret:
        print('Failed to grab frame')
        return 1

    # The frame will show with the title of the app
    cv2.imshow('2-Hand Pose Estimation - Screenshot', frame)

    # Wait for key to be pressed
    k = cv2.waitKey(1)

    # If the exit key is pressed, the app will stop
    if k == EXIT_KEY:
        print('Escape key pressed, closing the app')
        cam.release()

        return 1

    # If the continue key is pressed, screenshots will be taken
    elif k == CONTINUE_KEY:
        print(f'Screenshot taken and saved as {IMAGE_FP}')

        # Save screenshot and transform
        cv2.imwrite(IMAGE_FP, frame)
        img_transformation()

        # Close camera
        cam.release()
        cv2.destroyAllWindows()

        # Show predictions
        show_predictions()

        return 0

def show_predictions():

    # Load hand
    hand1 = DemoHand(
        file_name=IMAGE_FP,
        config=HANDPOSE_MODEL_CONFIG
    )

    # Load left-hand
    hand2 = DemoLeftHand(
        file_name=IMAGE_FP,
        estimator_config=HANDPOSE_MODEL_CONFIG,
        classifier_config=LEFT_RIGHT_MODEL_CONFIG
    )

    # Predictions
    img1 = hand1.pred_skeleton.resize(OUT_SIZE, Image.ANTIALIAS)
    opencv_img1 = cv2.cvtColor(np.array(img1), cv2.COLOR_BGR2RGB)

    img2 = hand2.pred_skeleton.resize(OUT_SIZE, Image.ANTIALIAS)
    opencv_img2 = cv2.cvtColor(np.array(img2), cv2.COLOR_BGR2RGB)

    # Display the image using OpenCV
    cv2.imshow('HandPoseEstimation', opencv_img1)
    cv2.imshow('HandPoseEstimation - v2', opencv_img2)

    # Display until continue key is pressed
    while True:
        k = cv2.waitKey(1)
        if k == CONTINUE_KEY:
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":

    while True:
        if take_screenshot() == 1:
            break

    # Delete image
    os.remove(IMAGE_FP)


