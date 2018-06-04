from eunet import EUnet
import torch.nn as nn


def get_segmentation_model():
    return EUnet(1, 4, 4, 1, 1, depth=2, padding=0, non_linearity=nn.ReLU, use_dropout=True, use_bn=True)