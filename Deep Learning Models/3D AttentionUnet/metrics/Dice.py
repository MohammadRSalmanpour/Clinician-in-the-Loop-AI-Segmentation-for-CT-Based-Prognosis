import torch

def DiceCoefficient(prediction, label, epsilon=1e-7):
    """
    Calculate the mean Dice score for a batch of predictions and labels.

    Args:
    prediction (torch.Tensor): Predicted tensor of shape (N, C, H, W)
    label (torch.Tensor): Ground truth tensor of the same shape as prediction
    epsilon (float): Smoothing factor to avoid division by zero

    Returns:
    torch.Tensor: Mean Dice score for the batch
    """
    # Calculate per-sample Dice coefficients
    dice_numerator = 2 * (prediction * label).sum(dim=[2, 3])
    dice_denominator = prediction.sum(dim=[2, 3]) + label.sum(dim=[2, 3])

    # Mean Dice score across the batch
    dice_score = (dice_numerator + epsilon) / (dice_denominator + epsilon)
    mean_dice_score = dice_score.mean()

    return mean_dice_score


