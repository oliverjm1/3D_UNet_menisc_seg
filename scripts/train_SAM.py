"""
Python script for training SAM.
This was not used in the end, and SAM was instead trained purely using a slice dataset.
The code for the train script used for SAM is found in train_SAM_slices.py.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import glob
import os
import copy
import sys
import wandb
from segment_anything import sam_model_registry
sys.path.append('../src')
from model_SAM import my_SAM
from metrics import dice_loss, dice_coefficient, batch_dice_coeff
from datasets import KneeSegDataset2DSAM
from utils import read_hyperparams, path_arr_to_slice_arr

# Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Read in hyperparams from txt file (will keep this in scripts folder)
# Each line in file in format (e.g. learning_rate=0.001)
hyperparams = read_hyperparams('hyperparams_sam.txt')
print(hyperparams)

# Define data path
DATA_DIR = '../data'

# Get the paths
train_paths = np.array([os.path.basename(i).split('.')[0] for i in glob.glob(f'{DATA_DIR}/train/*.im')])
val_paths = np.array([os.path.basename(i).split('.')[0] for i in glob.glob(f'{DATA_DIR}/valid/*.im')])

# Use function in utils to form train/val arrays of info needed to access a specific image slice
# Each element in the new array with contain the path to an image, along with the slice index [[path, index],...]
num_of_slices = 160

train_slice_array = path_arr_to_slice_arr(train_paths, num_of_slices)
val_slice_array = path_arr_to_slice_arr(val_paths, num_of_slices)

# Define the dataset and dataloaders
train_dataset = KneeSegDataset2DSAM(train_slice_array, DATA_DIR)
val_dataset = KneeSegDataset2DSAM(val_slice_array, DATA_DIR, split='valid')
train_loader = DataLoader(train_dataset, batch_size=int(hyperparams['batch_size']), num_workers = 1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, num_workers = 1, shuffle=False)

# Load in SAM with pretrained weights
sam_checkpoint = "../models/sam_vit_b_01ec64.pth"
model_type = "vit_b"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

# Create model, initialising with pretrained SAM parts
model = my_SAM(
    image_encoder=copy.deepcopy(sam.image_encoder),
    prompt_encoder=copy.deepcopy(sam.prompt_encoder),
    mask_decoder=copy.deepcopy(sam.mask_decoder),
)
model.eval()

# Specify optimiser
# Only use trainable parameters in optimiser
l_rate = hyperparams['l_rate']
trainable_params = [param for param in model.parameters() if param.requires_grad]
optimizer = optim.Adam(trainable_params, lr=l_rate)

# define bce loss. Will call this and dice loss in train loop, unweighted
loss_bce = nn.BCELoss()

# How long to train for?
num_epochs = int(hyperparams['num_epochs'])

# Threshold for predicted segmentation mask
threshold = hyperparams['threshold']

# start a new wandb run to track this script - LOG IN ON CONSOLE BEFORE RUNNING
wandb.init(
    # set the wandb project where this run will be logged
    project="train_SAM_model",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": l_rate,
    "architecture": "2D SAM",
    "unfrozen?": "Decoder",
    "dataset": "IWOAI",
    "epochs": num_epochs,
    "threshold": threshold,
    }
)

model.to(device)

# use multiple gpu in parallel if available
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# Train Loop
for epoch in range(num_epochs):

    # train mode
    model.train()
    running_loss = 0.0
    dice_coeff = 0.0
    n = 0    # counter for num of batches

    # Loop through train loader
    for idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward
        outputs = model(inputs)
        bce = loss_bce(outputs, targets) #binary cross-entropy
        dice = dice_loss(outputs, targets) #dice
        loss = bce + dice #unweighted combination of the two

        # Backward, and update params
        loss.backward()
        optimizer.step()

        running_loss += loss.detach().cpu().numpy()
        dice_coeff += batch_dice_coeff(outputs>threshold, targets).detach().cpu().numpy()
        n += 1

    # Get train metrics, averaged over number of images in batch
    train_loss = running_loss/n
    train_dice_av = dice_coeff/n

    # After each batch, loop through validation loader and get metrics
    # set model to eval mode and reset metrics
    model.eval()
    running_loss = 0.0
    dice_coeff = 0.0
    n = 0

    # Perform loop without computing gradients
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            bce = loss_bce(outputs, targets)
            dice = dice_loss(outputs, targets)
            loss = bce + dice

            running_loss += loss.detach().cpu().numpy()
            dice_coeff += batch_dice_coeff(outputs>threshold, targets).detach().cpu().numpy()
            n += 1

    # Val metrics
    val_loss = running_loss/n
    val_dice_av = dice_coeff/n
    
    # log to wandb
    wandb.log({"Train Loss": train_loss, "Train Dice Score": train_dice_av,
               "Val Loss": val_loss, "Val Dice Score": val_dice_av})
    
# Once training is done, save model
model_path = f"{hyperparams['run_name']}.pth"
torch.save(model.state_dict(), model_path)

wandb.finish()
