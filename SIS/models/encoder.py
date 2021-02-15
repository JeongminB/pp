import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, style_image):
        x = style_image
        return x