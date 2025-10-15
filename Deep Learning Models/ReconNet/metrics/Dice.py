import torch

def DiceCoefficient(label, prediction, smooth=1):
    """
    Computes the Dice Coefficient for 3D data.

    Args:
        label (torch.Tensor): Ground truth tensor of shape [batch, channel, depth, height, width].
        prediction (torch.Tensor): Predicted tensor of the same shape as label.
        smooth (float): Small constant to avoid division by zero.

    Returns:
        float: Mean Dice Coefficient across the batch.
    """
    # Flatten the tensors along all spatial dimensions
    label_flat = label.reshape(label.size(0), -1)  # [batch, total_voxels]
    prediction_flat = prediction.reshape(prediction.size(0), -1)

    # Calculate intersection and denominator
    intersection = torch.sum(label_flat * prediction_flat, dim=1)
    denominator = torch.sum(label_flat ** 2, dim=1) + torch.sum(prediction_flat ** 2, dim=1)

    # Compute Dice Coefficient for each batch element
    dice_score = (2. * intersection + smooth) / (denominator + smooth)

    # Average the Dice score over the batch dimension
    mean_dice_score = dice_score.mean()
    return mean_dice_score
