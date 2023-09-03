import os
from os import path

import cv2
import numpy as np
from PIL import Image

from demo.demo_model import DEMO_DIR, DemoHand, DemoLeftHand
from settings import HANDPOSE_MODEL_CONFIG, LEFT_RIGHT_MODEL_CONFIG, DATA

CONTINUE_KEY = 32  # space-bar
OUT_SIZE = (480, 480)
IMAGE_FP = path.join(DEMO_DIR, "img.jpg")
IN = path.join(DEMO_DIR, "in")

def show_predictions(file_name):

    file_name = path.join(IN, file_name)

    # Load hand
    hand1 = DemoHand(
        file_name=file_name,
        config=HANDPOSE_MODEL_CONFIG
    )

    # Load left-hand
    hand2 = DemoLeftHand(
        file_name=file_name,
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

    images = os.listdir(IN)
    images.sort()

    images_dict = {i+1: img for i, img in enumerate(images)}

    while True:

        print("------------ OPTIONS ------------")
        for i, img in images_dict.items():
            print(f" [{i}] {img}")
        print(f" [0] Exit")
        print("---------------------------------")

        choice = int(input("Select option: "))

        if choice == 0:
            break

        show_predictions(images_dict[choice])


