import numpy as np
from torch.nn import functional as F

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