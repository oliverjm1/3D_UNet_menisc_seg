"""
File containing utility functions.
Mostly to do with preprocessing of images before feeding to model.
E.g. clipping, cropping, and converting to SAM input format.
"""

import numpy as np
from torch.nn import functional as F
from torchvision.transforms.functional import resize
from segment_anything.utils.transforms import ResizeLongestSide

# Function to take MRI image, clip the pixel values to specified upper
# bound, and normalise between zero and this bound.
def clip_and_norm(image, upper_bound):
    # Clip intensity values
    image = np.clip(image, 0, upper_bound)

    # Normalize the image to the range [0, 1]
    norm = (image - 0) / (upper_bound - 0)

    return norm

# This function will crop the MRI images to a pre-chosen size.
# May alter to allow range as an argument.
def crop_im(image):
    dim1_lower, dim1_upper = 120, 320
    dim2_lower, dim2_upper = 70, 326

    cropped = image[dim1_lower:dim1_upper, dim2_lower:dim2_upper, :]

    return cropped

# This function will pad an image upto a square of a give size
def pad_to_square(x, size):
        h, w = x.shape[-2:]
        padh = size - h
        padw = size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

# Function that reads in txt file with each line in format x=y
# and converts to hyperparam dictionary
def read_hyperparams(path):
    hyperparams = {}
    with open(path, 'r') as file:
        for line in file:
            key, value = line.strip().split('=')
            # Convert to float if possible, else leave as string
            try:
                value = float(value)
            except ValueError:
                pass
            hyperparams[key] = value

    return hyperparams

# Function that transforms an image slice into the accepted input format for SAM
# My slices are 200x256 and SAM expects an rbg image 3x1024x1024
def sam_slice_transform(image):
    # Resizing, expanding channels, and padding to rgb 1024x1024
    # Make longest size 1024
    make_big = ResizeLongestSide(1024)

    target_size = make_big.get_preprocess_shape(
        image.shape[1], image.shape[2], make_big.target_length
    )

    big = resize(image, target_size, antialias=True)

    # Expand to 3 channels for RBG input
    rgb = big.repeat(3, 1, 1)

    # Pad to 1024x1024 square
    input = pad_to_square(rgb, 1024)
    return input


# Function that takes in an array of 3d image paths and the num of slices
# in each image, returning array of (path, slice_index) to access each slice.
def path_arr_to_slice_arr(path_array, num_of_slices):
    path_slice_array = []

    for path in path_array:
        for i in range(num_of_slices):
            path_slice_array.append((path, i))

    path_slice_array = np.array(path_slice_array)
    return path_slice_array