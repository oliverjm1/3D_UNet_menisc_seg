"""
This is the script used for fine-tuning SAM on the 2D slice dataset created from the 3D images/masks.
The SAM ViT-b encoder was used. SAM was initialised with the pre-trained model weights.

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
import random
from segment_anything import sam_model_registry
sys.path.append('../src')
from model_SAM import my_SAM
from metrics import dice_loss, dice_coefficient, batch_dice_coeff
from datasets import KneeSegDataset2DSlicesSAM
from utils import read_hyperparams

def main():
    # Set Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device', device)

    # Generate paramters randomly at the start of each run
    print("---- PARAMS ----")

    #batch size
    bs_list = [4,8,16]
    batch_size = random.choice(bs_list)
    print(f"batch size = {batch_size}")

    #learning rate
    lr_list = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4]
    l_rate = random.choice(lr_list)
    print(f"learning rate = {l_rate}")

    # Read in hyperparams from txt file (will keep this in scripts folder)
    # Each line in file in format (e.g. learning_rate=0.001)

    hyperparams = read_hyperparams('frozen_sam_sweep_params.txt')
    print(hyperparams)

    # Define data path
    DATA_DIR = '../data'

    # Get the paths
    train_paths = np.array([os.path.basename(i) for i in glob.glob(f'{DATA_DIR}/train_slice_ims/*')])
    val_paths = np.array([os.path.basename(i) for i in glob.glob(f'{DATA_DIR}/valid_slice_ims/*')])

    # Define the dataset and dataloaders
    train_dataset = KneeSegDataset2DSlicesSAM(train_paths, DATA_DIR)
    val_dataset = KneeSegDataset2DSlicesSAM(val_paths, DATA_DIR, split='valid')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers = 1, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers = 1, shuffle=False, pin_memory=True)

    print('trying dataloader')
    sys.stdout.flush()

    image2, mask2 = next(iter(val_loader))
    print("image2, mask2", image2, mask2)
    sys.stdout.flush()

    # Load in SAM with pretrained weights
    sam_checkpoint = "../models/sam_vit_b_01ec64.pth"
    model_type = "vit_b"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

    # Create model, initialising with pretrained SAM parts
    frozen_enc = (hyperparams['frozen_enc'] == "True")
    print(f"frozen enc? {frozen_enc}")

    model = my_SAM(
        image_encoder=copy.deepcopy(sam.image_encoder),
        prompt_encoder=copy.deepcopy(sam.prompt_encoder),
        mask_decoder=copy.deepcopy(sam.mask_decoder),
        freeze_encoder=frozen_enc,
    )
    
    # If continuing training from previous epoch save, 
    # get saved state dict and load
    # model_state_path = "test_sam.pth"
    # epochs_trained = 1
    # checkpoint = torch.load(model_state_path)
    # model.load_state_dict(checkpoint)
    epochs_trained = 0

    model.eval()

    # Specify optimiser
    # Only use trainable parameters in optimiser
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=l_rate)

    # define bce loss. Will call this and dice loss in train loop, unweighted
    loss_bce = nn.BCELoss()

    # How long to train for?
    num_epochs = int(hyperparams['num_epochs'])

    # threshold for assigning pixel probability to meniscus or background
    threshold = 0.5
    
    # OPTION FOR RUNNING WANDB OFFLINE
    # os.environ["WANDB_API_KEY"] = MY_API_KEY
    # os.environ["WANDB_MODE"] = "offline"

    # start a new wandb run to track this script - LOG IN ON CONSOLE BEFORE RUNNING
    wandb.init(
        # set the wandb project where this run will be logged
        project="train_SAM_JADE_runs",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": l_rate,
        "batch_size": batch_size,
        "architecture": "2D SAM",
        "unfrozen?": "All",
        "dataset": "IWOAI",
        "epochs": num_epochs,
        "threshold": threshold,
        }
    )

    model.to(device)

    print(f"CUDA COUNT: {torch.cuda.device_count()}")

    # # use multiple gpu in parallel if available
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    e_count = epochs_trained

    best_val_loss = 100

    # Train Loop
    for epoch in range(num_epochs):
        print('trains')
        # train mode
        model.train()
        running_loss = 0.0
        dice_coeff = 0.0
        
        n = 0    # counter for num of batches
        sys.stdout.flush()
        # Loop through train loader
        for inputs, targets in train_loader:
            sys.stdout.flush()

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
            for inputs, targets in val_loader:
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
        
        # print metrics
        print(f"Train Loss: {train_loss}, Train Dice Score: {train_dice_av},")
        print(f"Val Loss: {val_loss}, Val Dice Score: {val_dice_av}")

        # save as best if val loss is lowest so far
        if val_loss < best_val_loss:
            model_path = f"{hyperparams['run_name']}_best_E.pth"
            torch.save(model.state_dict(), model_path)
            print(f"best epoch yet: {epoch}")
            
            #reset best as current
            best_val_loss = val_loss

        e_count += 1
        
    # Once training is done, save model
    model_path = f"{hyperparams['run_name']}.pth"
    torch.save(model.state_dict(), model_path)

    wandb.finish()


if __name__ == '__main__':
    print('gonna do the main file')
    main()