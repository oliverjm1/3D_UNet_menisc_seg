"""
File containing utility functions for the skm-tea dataset.
These include transformations to resize and crop for model input/outputs.
Also functions for loading data such as train/val/test split indices
and pathology info.
"""

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from src.utils import crop_im, undo_crop

def skmtea_to_input_resize_crop(image) -> np.ndarray:
    """Function for transforming images from original skm-tea size 
    first to OAI size, and then cropping for unet input.

    Args:
        image (np.ndarray): skm-tea image or mask of size (512, 512, 160)

    Returns:
        np.ndarray: resized and cropped image of size (200, 256, 160)
    """

    # Resize from (512, 512, 160) to (384, 384, 160) using trilinear interpolation
    image_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 512, 512, 160)
    resized_tensor = F.interpolate(image_tensor, size=(384, 384, 160), mode='trilinear', align_corners=True)
    resized_image = resized_tensor.squeeze(0).squeeze(0).numpy() # Back to numpy

    # Crop and return
    cropped_image = crop_im(resized_image)

    return cropped_image

def output_to_skmtea_pad_resize(mask, as_numpy = False) -> np.ndarray:
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
