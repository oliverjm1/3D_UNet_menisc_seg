"""
Python script for performing a hyperparameter sweep of the 3D U-Net.
This was done using Weights and Biases (WandB).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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

# loss function?
criterion = bce_dice_loss

# Define sweep config
sweep_configuration = {
    "method": "random",
    "name": "sweep",
    "metric": {"goal": "maximize", "name": "Val Dice Score"},
    "parameters": {
        "batch_size": {"values": [1, 2, 4]},
        "epochs": {"values": [5]},
        "lr": {"values": [0.0005, 0.001, 0.002, 0.005, 0.01]},
        "threshold": {"values": [0.3, 0.5, 0.7]},
        "num_kernels": {"values": [8, 16, 32]},
    },
}

# Initialize sweep by passing in config.
# Provide a name of the project.
sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-unet-sweep")

def train_epoch(model, loader, optimizer, threshold):

    # train mode
    model.train()
    running_loss = 0.0
    dice_coeff = 0.0
    n = 0    # counter for num of batches

    # Loop through train loader
    for (inputs, targets) in loader:
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
    train_loss = running_loss/n
    train_dice = dice_coeff/n

    return train_loss, train_dice

def val_epoch(model, loader, threshold):
    
    # set model to eval mode
    model.eval()
    running_loss = 0.0
    dice_coeff = 0.0
    n = 0

    # Perform loop without computing gradients
    with torch.no_grad():
        for (inputs, targets) in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.detach().cpu().numpy()
            dice_coeff += batch_dice_coeff(outputs>threshold, targets).detach().cpu().numpy()
            n += 1

    # Val metrics
    val_loss = running_loss/n
    val_dice = dice_coeff/n

    return val_loss, val_dice

def main():

    run = wandb.init()

    # Define data path
    DATA_DIR = '../data'

    # Get the paths
    train_paths = np.array([os.path.basename(i).split('.')[0] for i in glob.glob(f'{DATA_DIR}/train/*.im')])
    val_paths = np.array([os.path.basename(i).split('.')[0] for i in glob.glob(f'{DATA_DIR}/valid/*.im')])

    # Define the dataset and dataloaders
    train_dataset = KneeSegDataset3D(train_paths, DATA_DIR)
    val_dataset = KneeSegDataset3D(val_paths, DATA_DIR, split='valid')
    train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, num_workers = 1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, num_workers = 1, shuffle=False)

    # Create model
    model = UNet3D(1, 1, wandb.config.num_kernels)

    # Specify optimiser
    l_rate = wandb.config.lr
    optimizer = optim.Adam(model.parameters(), lr=l_rate)

    # How long to train for?
    num_epochs = wandb.config.epochs

    # Threshold for predicted segmentation mask
    threshold = wandb.config.threshold

    model.to(device)

    # use multiple gpu in parallel if available
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Epoch Loop
    for epoch in range(num_epochs):

        # Train loop
        train_loss, train_dice = train_epoch(model, train_loader, optimizer, threshold)

        # Validation loop
        val_loss, val_dice = val_epoch(model, val_loader, threshold)
        
        # log to wandb
        wandb.log(
            {
                "Train Loss": train_loss, 
                "Train Dice Score": train_dice,
                "Val Loss": val_loss, 
                "Val Dice Score": val_dice,
            }
        )

# Start sweep job.
wandb.agent(sweep_id, function=main, count= 10)

