import torch
import torch.nn as nn

# Define Dice Coefficient for two masks
def dice_coefficient(mask1, mask2):
    intersection = torch.sum(mask1 * mask2)
    sum = mask1.sum().item() + mask2.sum().item()
    dice = (2.0 * intersection) / sum
    return dice.item()

# Return average dice coeff for a batch of input and target masks
# 'smooth' constant included to avoid NaN errors when volume is zero
def batch_dice_coeff(input, target, smooth=1e-5):
    # get tuple of spatial dimensions (first two dims are batch and channel)
    spatial_dims = tuple(range(2, len(input.shape)))
    
    # calculate intersection & sum of masks, then calculate dice coeff
    intersection = torch.sum(input * target, dim=spatial_dims)
    sum = torch.sum(input, dim=spatial_dims) + torch.sum(target, dim=spatial_dims)
    dice = (2.0 * intersection + smooth) / (sum + smooth)

    # return mean dice coeff of batch
    return torch.mean(dice)

# Define Dice Loss
# This is 1 - dice coeff.
def dice_loss(input, target, smooth=1e-5):
    mean_loss = 1 - batch_dice_coeff(input, target, smooth=smooth)
    return mean_loss

# Loss that includes both binary cross-entropy and dice loss
def bce_dice_loss(outputs, targets):
    dice = dice_loss(outputs, targets)
    bceloss = nn.BCELoss()
    bce = bceloss(outputs, targets)
    return bce + dice