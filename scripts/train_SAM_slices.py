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
from datasets import KneeSegDataset2DSlicesSAM
from utils import read_hyperparams

def main():
    print('here')

    # Set Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device', device)

    # Read in hyperparams from txt file (will keep this in scripts folder)
    # Each line in file in format (e.g. learning_rate=0.001)
    hyperparams = read_hyperparams('hyperparams_sam.txt')
    print(hyperparams)

    # Define data path
    DATA_DIR = '../data'

    # Get the paths
    train_paths = np.array([os.path.basename(i) for i in glob.glob(f'{DATA_DIR}/train_slice_ims/*')])
    val_paths = np.array([os.path.basename(i) for i in glob.glob(f'{DATA_DIR}/valid_slice_ims/*')])

    # Define the dataset and dataloaders
    train_dataset = KneeSegDataset2DSlicesSAM(train_paths, DATA_DIR)
    val_dataset = KneeSegDataset2DSlicesSAM(val_paths, DATA_DIR, split='valid')
    train_loader = DataLoader(train_dataset, batch_size=int(hyperparams['batch_size']), num_workers = 0, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers = 0, shuffle=False, pin_memory=True)

    print('dataloaders defined')

    print('test getting an item')
    image, mask = val_dataset.__getitem__(0)
    print(image.shape)
    print(mask.shape)
    print('image, mask', image, mask)

    print('trying dataloader')
    image2, mask2 = next(iter(val_loader))
    print("image2, mask2", image2, mask2)

    print('trying dataloader iterate')
    for idx, (image2, mask2) in enumerate(val_loader):
        print("load ",idx)

    # # Load in SAM with pretrained weights
    # sam_checkpoint = "../models/sam_vit_b_01ec64.pth"
    # model_type = "vit_b"

    # sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

    # # Create model, initialising with pretrained SAM parts
    # model = my_SAM(
    #     image_encoder=copy.deepcopy(sam.image_encoder),
    #     prompt_encoder=copy.deepcopy(sam.prompt_encoder),
    #     mask_decoder=copy.deepcopy(sam.mask_decoder),
    # )
    # model.eval()

    # # Specify optimiser
    # # Only use trainable parameters in optimiser
    # l_rate = hyperparams['l_rate']
    # trainable_params = [param for param in model.parameters() if param.requires_grad]
    # optimizer = optim.Adam(trainable_params, lr=l_rate)

    # # define bce loss. Will call this and dice loss in train loop, unweighted
    # loss_bce = nn.BCELoss()

    # # How long to train for?
    # num_epochs = int(hyperparams['num_epochs'])

    # # Threshold for predicted segmentation mask
    # threshold = hyperparams['threshold']

    # # start a new wandb run to track this script - LOG IN ON CONSOLE BEFORE RUNNING
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="train_SAM_model",
        
    #     # track hyperparameters and run metadata
    #     config={
    #     "learning_rate": l_rate,
    #     "architecture": "2D SAM",
    #     "unfrozen?": "Decoder",
    #     "dataset": "IWOAI",
    #     "epochs": num_epochs,
    #     "threshold": threshold,
    #     }
    # )

    # model.to(device)

    # # # use multiple gpu in parallel if available
    # # if torch.cuda.device_count() > 1:
    # #     model = nn.DataParallel(model)

    # # Train Loop
    # for epoch in range(num_epochs):
    #     print('trains')
    #     # train mode
    #     model.train()
    #     running_loss = 0.0
    #     dice_coeff = 0.0
    #     n = 0    # counter for num of batches

    #     # Loop through train loader
    #     for idx, (inputs, targets) in enumerate(train_loader):
    #         print("in loader")

    #         inputs = inputs.to(device)
    #         targets = targets.to(device)

    #         optimizer.zero_grad()

    #         # Forward
    #         outputs = model(inputs)
    #         bce = loss_bce(outputs, targets) #binary cross-entropy
    #         dice = dice_loss(outputs, targets) #dice
    #         loss = bce + dice #unweighted combination of the two

    #         # Backward, and update params
    #         loss.backward()
    #         optimizer.step()

    #         running_loss += loss.detach().cpu().numpy()
    #         dice_coeff += batch_dice_coeff(outputs>threshold, targets).detach().cpu().numpy()
    #         n += 1
    #         print(idx)

    #     # Get train metrics, averaged over number of images in batch
    #     train_loss = running_loss/n
    #     train_dice_av = dice_coeff/n

    #     # After each batch, loop through validation loader and get metrics
    #     # set model to eval mode and reset metrics
    #     model.eval()
    #     running_loss = 0.0
    #     dice_coeff = 0.0
    #     n = 0

    #     # Perform loop without computing gradients
    #     with torch.no_grad():
    #         for idx, (inputs, targets) in enumerate(val_loader):
    #             inputs = inputs.to(device)
    #             targets = targets.to(device)

    #             outputs = model(inputs)
    #             bce = loss_bce(outputs, targets)
    #             dice = dice_loss(outputs, targets)
    #             loss = bce + dice

    #             running_loss += loss.detach().cpu().numpy()
    #             dice_coeff += batch_dice_coeff(outputs>threshold, targets).detach().cpu().numpy()
    #             n += 1

    #     # Val metrics
    #     val_loss = running_loss/n
    #     val_dice_av = dice_coeff/n
        
    #     # log to wandb
    #     wandb.log({"Train Loss": train_loss, "Train Dice Score": train_dice_av,
    #             "Val Loss": val_loss, "Val Dice Score": val_dice_av})
        
    # # Once training is done, save model
    # model_path = f"{hyperparams['run_name']}.pth"
    # torch.save(model.state_dict(), model_path)

    # wandb.finish()


if __name__ == '__main__':
    print('gonna do the main file')
    main()