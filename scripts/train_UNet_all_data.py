"""
Python script for training the 3D U-Net on all data (train and validation) post hyperparameter tuning.
A .txt file containing hyperparameter values is read in at the start.
This allowed for altering hyperparameters on ARC without editing the .py file.
Weights and Biases was used to track runs.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import glob
import os
import sys
import wandb
sys.path.append('../src')
from model_UNet import UNet3D
from metrics import bce_dice_loss, dice_coefficient, batch_dice_coeff
from datasets import KneeSegDataset3D
from utils import read_hyperparams

# Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Read in hyperparams from txt file (will keep this in scripts folder)
# Each line in file in format (e.g. learning_rate=0.001)
hyperparams = read_hyperparams('hyperparams_unet_final.txt')
print(hyperparams)

# Define data path
DATA_DIR = '../data'

# Get the paths
train_paths = np.array([os.path.basename(i).split('.')[0] for i in glob.glob(f'{DATA_DIR}/train/*.im')])
val_paths = np.array([os.path.basename(i).split('.')[0] for i in glob.glob(f'{DATA_DIR}/valid/*.im')])

# Define the dataset and dataloaders
train_dataset = KneeSegDataset3D(train_paths, DATA_DIR)
val_dataset = KneeSegDataset3D(val_paths, DATA_DIR, split='valid')

# Combine the datasets
train_val_dataset = ConcatDataset([train_dataset,val_dataset])
loader = DataLoader(train_val_dataset, batch_size=int(hyperparams['batch_size']), num_workers = 1, shuffle=True)

# Create model
model = UNet3D(1, 1, 16)

# Specify optimiser and criterion
criterion = bce_dice_loss
l_rate = hyperparams['l_rate']
optimizer = optim.Adam(model.parameters(), lr=l_rate)

# How long to train for?
num_epochs = int(hyperparams['num_epochs'])

# Threshold for predicted segmentation mask
threshold = hyperparams['threshold']

# start a new wandb run to track this script - LOG IN ON CONSOLE BEFORE RUNNING
wandb.init(
    # set the wandb project where this run will be logged
    project="train_seg_model_all_data",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": l_rate,
    "architecture": "3D UNet",
    "kernel_num": 16,
    "dataset": "IWOAI",
    "epochs": num_epochs,
    "threshold": threshold,
    }
)

model.to(device)

# use multiple gpu in parallel if available
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

best_loss = 100

# Train Loop
for epoch in range(num_epochs):

    # train mode
    model.train()
    running_loss = 0.0
    dice_coeff = 0.0
    n = 0    # counter for num of batches

    # Loop through train loader
    for idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward, backward, and update params
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.detach().cpu().numpy()
        dice_coeff += batch_dice_coeff(outputs>threshold, targets).detach().cpu().numpy()
        n += 1

    # Get train metrics, averaged over number of images in batch
    loss = running_loss/n
    dice_av = dice_coeff/n

    # After each batch, loop through validation loader and get metrics
    # set model to eval mode and reset metrics
    model.eval()
    running_loss = 0.0
    dice_coeff = 0.0
    n = 0
    
    # log to wandb
    wandb.log({"Loss": loss, "Dice Score": dice_av})
    
    # save as best if val loss is lowest so far
    if loss < best_loss:
        model_path = f"{hyperparams['run_name']}_best_E.pth"
        torch.save(model.state_dict(), model_path)
        print(f"best epoch yet: {epoch}")
        
        #reset best as current
        best_loss = loss
    
# Once training is done, save model
model_path = f"{hyperparams['run_name']}.pth"
torch.save(model.state_dict(), model_path)

wandb.finish()
