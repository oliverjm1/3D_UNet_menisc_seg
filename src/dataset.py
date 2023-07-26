import torch
from torch.utils.data import Dataset
import h5py
import os
import numpy as np
from utils import crop_im, clip_and_norm

# Define the Dataset class
class KneeSegDataset(Dataset):
    def __init__(self, file_paths, data_dir, split='train'):
        self.file_paths = file_paths
        self.data_dir = data_dir
        self.split = split

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        path = self.file_paths[index]

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
        mask = np.clip(minisc_mask, 0, 1)


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