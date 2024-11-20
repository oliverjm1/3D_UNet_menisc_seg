"""
File containing utility functions for the skm-tea dataset.
These include transformations to resize and crop for model input/outputs.
Also functions for loading data such as train/val/test split indices
and pathology info.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import os
import h5py
from typing import Tuple
from torch.nn import functional as F
from src.utils import crop_im, undo_crop, clip_and_norm

def skmtea_to_input_resize_crop(image) -> np.ndarray:
    """Function for transforming images from original skm-tea size 
    first to OAI size, and then cropping for unet input.

    Args:
        image (np.ndarray): skm-tea image or mask of size (512, 512, 160)

    Returns:
        np.ndarray: resized and cropped image of size (200, 256, 160)
    """

    # Resize from (512, 512, 160) to (384, 384, 160) using trilinear interpolation
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 512, 512, 160)

    # check if input is binary - will need to use nearest interpolation
    if len(np.unique(image)) == 2:
        resized_tensor = F.interpolate(image_tensor, size=(384, 384, 160), mode='nearest')
        resized_tensor = (resized_tensor > 0.5).int() # Convert back to binary
    else:
        resized_tensor = F.interpolate(image_tensor, size=(384, 384, 160), mode='trilinear', align_corners=True)

    resized_image = resized_tensor.squeeze(0).squeeze(0).numpy() # Back to numpy

    # Crop and return
    cropped_image = crop_im(resized_image)

    return cropped_image

def output_to_skmtea_pad_resize(mask, as_numpy = False):
    """Function to transform output masks back to original skm-tea size.

    Args:
        image (np.ndarray): output masks of size (200, 256, 160)
        as_numpy (bool, optional): convert to numpy array or keep as torch tensor. Defaults to False.

    Returns:
        torch.dtype OR np.ndarray: if tensor, size (1, 1, 512, 512, 160)
        if numpy array, size (512, 512, 160)
    """

    # Pad back to (384, 384, 160)
    padded_mask = undo_crop(mask)

    # Resize up to (512, 512, 160)
    mask_tensor = torch.tensor(padded_mask).unsqueeze(0).unsqueeze(0).float()  # Shape: (1, 1, 384, 384, 160)
    resized_tensor = F.interpolate(mask_tensor, size=(512, 512, 160), mode='trilinear', align_corners=True).round()
    
    # Convert to numpy if specified
    if as_numpy:
        resized_image = resized_tensor.squeeze(0).squeeze(0).numpy()
    else:
        resized_image = resized_tensor

    return resized_image

def echo_combination(echo1: np.ndarray, echo2: np.ndarray):
    """Function to combine the two echos into a final image.
    I DO NOT YET KNOW BEST WAY OF DOING THIS. CURRENTLY DOING ROOT SUM OF SQUARES.

    Returns:
        np.ndarray: combination of the two echos
    """

    # Normalise echos to between 0 and 1
    norm_echo1 = clip_and_norm(echo1)
    norm_echo2 = clip_and_norm(echo2)

    # Compute the RSS of the two rescaled echos
    rss = np.sqrt(norm_echo1**2 + norm_echo2**2)

    # clip and rescale rss image based on looking at histogram of signal
    combined_image = clip_and_norm(rss, 0.6)

    return combined_image

def get_skmtea_im_and_seg(file_path: str, data_dir: str, only_menisci = True) -> Tuple[np.ndarray, np.ndarray]:
    """Function to return image and segmentation masks of a given skm-tea image.

    Args:
        file_path (str): image file name
        data_dir (str): path to folder the image is in
        only_menisci (bool, optional): Return just menisci or all structure masks.
            Defaults to True.

    Returns:
        tuple of np.ndarray: Combined echo image and segmentation masks,
            either of just menisci or of all cartilage.
    """
    full_path = os.path.join(data_dir, file_path)

    # get full paths and read in
    # Open the HDF5 file in read mode
    with h5py.File(full_path, 'r') as hf:
        # Load Echo 1 and Echo 2 data
        echo1 = hf['echo1'][:].astype(np.float64)
        echo2 = hf['echo2'][:].astype(np.float64)

        # Load segmentation data (One-hot encoded, 6 classes)
        seg = hf['seg'][:]

    image = echo_combination(echo1, echo2)

    if only_menisci:
        # menisci
        med_mask = seg[...,4]
        lat_mask = seg[...,5]

        # combine
        seg = np.add(med_mask, lat_mask)

    return (image, seg)

def plot_with_bbox(image, bbox, slice_idx, leeway=20, savefig=False, savepath=None):
    """
    Plots an MRI slice with a bounding box and a zoomed-in view of the region around the box.
    
    Parameters:
    - image: 3D numpy array representing the MRI volume (height, width, slices).
    - bbox: Tuple (y_min, x_min, z_min, width, height, depth) defining the bounding box.
    - slice_idx: Integer index for the slice to be visualized.
    - leeway: Additional padding around the bounding box for the zoomed-in region.
    """
    # Extract bounding box info in x and y directions
    y_min, x_min, _, height, width, _ = bbox
    
    # Original MRI slice with bounding box
    plt.figure(figsize=(12, 6))
    
    # Full-size plot
    plt.subplot(1, 2, 1)
    plt.imshow(image[:, :, slice_idx], cmap='gray')
    plt.title("MRI with Overlaid Bounding Box")
    rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='red', facecolor='none')
    plt.gca().add_patch(rect)
    plt.axis('off')
    
    # Define the zoomed-in region
    x_min_zoom = max(0, int(x_min - leeway))
    y_min_zoom = max(0, int(y_min - leeway))
    x_max_zoom = min(image.shape[1], int(x_min + width + leeway))
    y_max_zoom = min(image.shape[0], int(y_min + height + leeway))
    
    # Extract the zoomed-in region
    zoomed_roi = image[y_min_zoom:y_max_zoom, x_min_zoom:x_max_zoom, slice_idx]
    
    # Zoomed-in plot
    plt.subplot(1, 2, 2)
    plt.imshow(zoomed_roi, cmap='gray')
    plt.title("Zoomed-in ROI Around Bounding Box")
    plt.axis('off')
    
    # Add the bounding box to the zoomed-in plot
    zoomed_rect = patches.Rectangle((leeway, leeway), width, height,
                                    linewidth=2, edgecolor='red', facecolor='none')
    plt.gca().add_patch(zoomed_rect)
    
    plt.tight_layout()

    # make sure savefig is True AND savepath is provided
    if savefig and savepath is not None:
        plt.savefig(savepath)
    
    plt.show()