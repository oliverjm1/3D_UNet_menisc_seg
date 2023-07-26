import torch
import torch.nn as nn

# Define Dice Coefficient for two masks
def dice_coefficient(mask1, mask2):
    intersection = torch.sum(mask1 * mask2)
    sum = mask1.sum().item() + mask2.sum().item()
    dice = (2.0 * intersection) / sum
    return dice.item()

# Return average dice coeff for a batch of input and target masks
def batch_dice_coeff(outputs, targets):
    # inputs/targets both have 5 dims (spatial are final 3)
    intersection = torch.sum(outputs * targets, dim=(2,3,4))
    sum = torch.sum(outputs, dim=(2,3,4)) + torch.sum(targets, dim=(2,3,4))
    dice = (2.0 * intersection) / sum
    return torch.mean(dice)

# Define Dice Loss
# This is 1 - dice coeff. Need to be wary of batch/channel dims.
# Calculate dice loss for each item in batch and average.
def dice_loss(outputs, targets):
    # inputs/targets both have 5 dims (spatial are final 3)
    intersection = torch.sum(outputs * targets, dim=(2,3,4))
    sum = torch.sum(outputs, dim=(2,3,4)) + torch.sum(targets, dim=(2,3,4))
    dice = (2.0 * intersection) / sum
    loss = 1 - dice
    return torch.mean(loss)

# Loss that includes both binary cross-entropy and dice loss
def bce_dice_loss(outputs, targets):
    dice = dice_loss(outputs, targets)
    bceloss = nn.BCELoss()
    bce = bceloss(outputs, targets)
    return bce + dice