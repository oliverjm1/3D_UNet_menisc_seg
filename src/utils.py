import numpy as np

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