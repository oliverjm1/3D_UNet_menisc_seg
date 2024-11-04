"""
File where all datasets are defined.
For the 3D U-Net, the 3D dataset class was used, where the full images and masks were loaded.
For training SAM, the 3D images/masks were all split up into slices and saved as 2D images. 
These were then loaded in using the KneeSegDataset2DSlicesSAM dataset.
"""

import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import h5py
import os
import numpy as np
from src.utils import crop_im, clip_and_norm, pad_to_square, sam_slice_transform


# Define the 3D Dataset class
# Image and mask both need same transforms to be applied, so DO NOT USE RANDOM TRANSFORMS
# - use e.g. transforms.functional.hflip which has no randomness.
class KneeSegDataset3D(Dataset):
    def __init__(self, file_paths, data_dir, split='train', transform=None, transform_chance=0.5):
        self.file_paths = file_paths
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.transform_chance = transform_chance

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        path = self.file_paths[index]

        # Test data is arranged differently, and as mask is numpy as opposed to h5py
        if self.split == 'test':
            im_path = os.path.join(self.data_dir, self.split, path + '.im')
            seg_path = os.path.join(self.data_dir, 'ground-truth', path + '.npy')
            with h5py.File(im_path,'r') as hf:
                image = np.array(hf['data'])
            mask = np.load(seg_path)

        # Train and Validation h5py files
        else: 
            # get full paths and read in
            im_path = os.path.join(self.data_dir, self.split, path + '.im')
            seg_path = os.path.join(self.data_dir, self.split, path + '.seg')
            with h5py.File(im_path,'r') as hf:
                image = np.array(hf['data'])
            with h5py.File(seg_path,'r') as hf:
                mask = np.array(hf['data'])

        if self.split == 'test':
            minisc_mask = mask[...,-1]
        else:

            #medial meniscus
            med_mask = mask[...,-1]

            # THERE IS ONE ERRANT CASE IN TRAIN SET. LATERAL MENISCUS IS AT WRONG INDEX
            #lateral
            if path == 'train_026_V01':
                lat_mask = mask[...,2]
            else:
                lat_mask = mask[...,-2]

            #both together
            minisc_mask = np.add(med_mask,lat_mask)

        mask = np.clip(minisc_mask, 0, 1) #just incase the two menisci ground truths overlap, clip at 1

        # crop image/mask
        image = crop_im(image)
        mask = crop_im(mask)

        # normalise image
        image = clip_and_norm(image, 0.005)

        # turn to torch, add channel dimension, and return
        image = torch.from_numpy(image).float().unsqueeze(0)
        mask = torch.from_numpy(mask).float().unsqueeze(0)

        # transforms?
        if self.transform != None:

            # Need to take care of randomness myself because need same transform applied to image and gt
            if random.random() < self.transform_chance:
                image = self.transform(image)
                mask = self.transform(mask)

                print("We flipping")

        return image, mask

# Dataset that opens image slice file, prepares for SAM input, and returns image and mask
class KneeSegDataset2DSlicesSAM(Dataset):
    def __init__(self, paths, data_dir, split='train'):
        self.paths = paths
        self.split = split
        self.data_dir = data_dir

        # set image and mask dir based on the split
        self.im_dir = f"{self.split}_slice_ims"
        self.mask_dir = f"{self.split}_slice_gts"

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]

        # load slice and mask
        im_path = os.path.join(self.data_dir, self.im_dir, path)
        mask_path = os.path.join(self.data_dir, self.mask_dir, path)
        image = np.load(im_path)
        mask = np.load(mask_path)

        # turn to torch, add channel dimension
        image = torch.from_numpy(image).float().unsqueeze(0)
        mask = torch.from_numpy(mask).float().unsqueeze(0)

        # ------ SAM STUFF ------
        input = sam_slice_transform(image)

        return input, mask

# Dataset class for segmentation task of SKM-TEA data.
# Return image and corresponding meniscus mask ground truth.
# Make sure that image has been rescaled to match OAI data (384,384,160).
# Then perform same cropping as done with training data.
class SKMTEASegDataset(Dataset):
    def __init__(self, file_paths, data_dir):
        self.file_paths = file_paths
        self.data_dir = data_dir

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        file_path = self.file_paths[index]

        full_path = os.path.join(self.data_dir, file_path)

        # get full paths and read in
        # Open the HDF5 file in read mode
        with h5py.File(full_path, 'r') as hf:
            # Load Echo 1 and Echo 2 data
            echo1 = hf['echo1'][:].astype(np.float64)
            echo2 = hf['echo2'][:].astype(np.float64)
    
            # Load segmentation data (One-hot encoded, 6 classes)
            seg = hf['seg'][:]

        # combine echos in some way... DECIDE BEST WAY - CLIP USING PERCENTILES -> HISTOGRAM NORM??
        # Normalise echos to between 0 and 1
        norm_echo1 = clip_and_norm(echo1)
        norm_echo2 = clip_and_norm(echo2)

        # Compute the RSS of the two rescaled echos
        rss = np.sqrt(norm_echo1**2 + norm_echo2**2)

        # clip and rescale rss image
        image = clip_and_norm(rss, 0.6)

        # menisci
        med_mask = seg[...,4]

        lat_mask = seg[...,5]

        # combine
        minisc_mask = np.add(med_mask, lat_mask)

        mask = np.clip(minisc_mask, 0, 1) #just incase the two menisci ground truths overlap, clip at 1

        # Resize image to match OAI (384,384,160)
        # Convert the 3D image to a PyTorch tensor
        image_tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 512, 512, 160)
        # Resize the image using trilinear interpolation
        resized_tensor = F.interpolate(image_tensor, size=(384, 384, 160), mode='trilinear', align_corners=True)
        # Back to numpy
        resized_image = resized_tensor.squeeze(0).squeeze(0).numpy()

        # crop image (unsure on mask yet)
        image = crop_im(resized_image)
        #mask = crop_im(mask)

        # turn to torch, add channel dimension, and return
        image = torch.from_numpy(image).float().unsqueeze(0)
        mask = torch.from_numpy(mask).float().unsqueeze(0)

        # transforms?

        return image, mask
