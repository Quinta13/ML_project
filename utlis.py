"""
This file provides some utilities...
"""

import matplotlib.pyplot as plt
import numpy as np
import torch


def from_image_to_heatmap(hand, model):
    img = hand.image_arr_z
    img_transposed = np.transpose(img, axes=(2, 0, 1))
    img_tensor = torch.from_numpy(img_transposed).unsqueeze(0)

    pred_heatmaps = model(img_tensor)[0].detach().numpy()

    return pred_heatmaps


def pad_idx(idx: int) -> str:
    return str(idx).zfill(8)


def plot_multiple_images(images):
    # Set the number of columns for displaying the images
    num_columns = 6

    # Calculate the number of rows needed based on the number of columns and images
    num_images = len(images)
    num_rows = (num_images + num_columns - 1) // num_columns

    # Create a figure with subplots arranged in a row
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(num_columns * 4, num_rows * 4))

    # Flatten the axes array to simplify indexing
    axes = axes.flatten()

    # Loop through the images and display them in the subplots
    for i, img in enumerate(images):
        # Display the image in the corresponding subplot
        axes[i].imshow(img)
        axes[i].axis('off')

    # Hide any remaining empty subplots (if the number of images is not a multiple of num_columns)
    for i in range(num_images, num_rows * num_columns):
        axes[i].axis('off')

    # Adjust the layout of the plots to avoid overlapping
    plt.tight_layout()

    # Display the images in a row
    plt.show()
