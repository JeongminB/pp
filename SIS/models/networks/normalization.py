"""
original paper: https://github.com/NVlabs/SPADE
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# spatially-adaptive (de)normalization (https://arxiv.org/abs/1903.07291)
# norm_ch: the #channels of the input features
# mask_ch: the #channels of the segmantation mask
# hidden_ch: the channel dimension of the embedding space which mask is projected onto
class SPADE(nn.Module):
    def __init__(self, norm_ch, mask_ch, 
            hidden_ch=128, kernel_size=3, stride_size=1, pad_size=1, 
            norm='batch', act='relu'):
        super().__init__()

        # the parameter-free normalization layer for input activation
        if norm == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_ch, affine=False)
        elif norm == 'syncbatch':
            self.param_free_norm = nn.SyncBatchNorm(norm_ch, affine=False)  # input: mini-batch of [N-2]D inputs with C-channels

        # the conv layers for the input segmentation mask
        self.conv_share = nn.Conv2d(mask_ch, hidden_ch, kernel_size, stride_size, pad_size)
        self.conv_beta = nn.Conv2d(hidden_ch, norm_ch, kernel_size, stride_size, pad_size)
        self.conv_gamma = nn.Conv2d(hidden_ch, norm_ch, kernel_size, stride_size, pad_size)
        
        # the activation layer of the conv_share
        if act == 'relu':
            self.conv_act = nn.ReLU()
        elif act == 'leaky_relu':
            self.conv_act = nn.LeakyReLU(0.2)


    def forward(self, x, mask):
        normalized_x = self.param_free_norm(x)

        # compute affine parameters
        m = F.interpolate(mask, size=x.size()[2:], mode='nearest')
        embedded_m = self.conv_share(m)
        gamma = self.conv_gamma(embedded_m)
        beta = self.conv_beta(embedded_m)

        output = (1 + gamma) * normalized_x + beta  # denormalize
        return output
