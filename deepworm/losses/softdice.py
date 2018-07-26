import torch


def soft_dice_loss(y_pred, y_true, epsilon=1e-6):
    ''' 
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format.

    # Arguments
        y_pred: b x N x X x Y Network output, must sum to 1 over c channel (such as after softmax) 
        y_true: b x N x X x Y  One hot encoding of ground truth       
        epsilon: Used for numerical stability to avoid divide by zero errors

    # References
        V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation 
        https://arxiv.org/abs/1606.04797
        More details on Dice loss formulation 
        https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)

        Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
    '''

    # skip the batch and class axis for calculating Dice score
    axes = tuple(range(1, len(y_pred.shape) - 1))
    numerator = 2. * torch.sum(y_pred * y_true, axes)
    denominator = torch.sum(y_pred.pow(2) + y_true, axes)

    return 1 - torch.mean(numerator / (denominator + epsilon))  # average over classes and batch
