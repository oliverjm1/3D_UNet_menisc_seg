import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import h5py
import os
import numpy as np
from utils import crop_im, clip_and_norm, pad_to_square, sam_slice_transform


# Define the 3D Dataset class
class KneeSegDataset3D(Dataset):
    def __init__(self, file_paths, data_dir, split='train'):
        self.file_paths = file_paths
        self.data_dir = data_dir
        self.split = split

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

        #medial meniscus
        med_mask = mask[...,-1]

        #lateral
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

        # Do transforms here later

        return image, mask
    
# define dataset that will return a slice of an image ready for input into SAM
"""class KneeSegDataset2DSAM(Dataset):
    def __init__(self, slice_paths, data_dir, split='train'):
        self.slice_paths = slice_paths
        self.data_dir = data_dir
        self.split = split

    def __len__(self):
        return len(self.slice_paths)

    def __getitem__(self, index):
        path_and_slice_num = self.slice_paths[index]

        # extract the path and slice index from the array element
        path = path_and_slice_num[0]
        slice_num = int(path_and_slice_num[1])

        # get full paths and read in whole image
        im_path = os.path.join(self.data_dir, self.split, path + '.im')
        seg_path = os.path.join(self.data_dir, self.split, path + '.seg')
        with h5py.File(im_path,'r') as hf:
            image = np.array(hf['data'])
        with h5py.File(seg_path,'r') as hf:
            mask = np.array(hf['data'])

        #medial meniscus
        med_mask = mask[...,-1]

        #lateral
        lat_mask = mask[...,-2]

        #both together
        minisc_mask = np.add(med_mask,lat_mask)
        mask = np.clip(minisc_mask, 0, 1)

        # crop image/mask
        image = crop_im(image)
        mask = crop_im(mask)

        # normalise image
        image = clip_and_norm(image, 0.005)

        # turn to torch, add channel dimension
        image = torch.from_numpy(image).float().unsqueeze(0)
        mask = torch.from_numpy(mask).float().unsqueeze(0)

        # ------ SAM STUFF ------
        # Get one slice using the index
        im_slice = image[...,slice_num]
        mask_slice = mask[...,slice_num]

        # Resizing, expanding channels, and padding to rgb 1024x1024
        # Make longest size 1024
        make_big = ResizeLongestSide(1024)
        target_size = make_big.get_preprocess_shape(
            im_slice.shape[1], im_slice.shape[2], make_big.target_length
        )
        big_slice = resize(im_slice, target_size, antialias=True)

        # Expand to 3 channels for RBG input
        expand_dims = transforms.Lambda(lambda x: x.expand(3, -1, -1)) 
        rgb_slice = expand_dims(big_slice)
        
        # Pad to 1024x1024 square
        input_slice = pad_to_square(rgb_slice, 1024)

        return input_slice, mask_slice"""


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