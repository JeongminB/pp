"""
original paper: https://github.com/NVlabs/SPADE
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork


class ImageEncoder(BaseNetwork):
    def __init__(self, params):
        super().__init__()
        self.act = nn.LeakyReLU(0.2)

        kw = 3
        pw = int(np.ceil((kw-1.)/2))
        nef = params.nef

        self.layer1 = self.get_norm_layer(nn.Conv2d(3, nef, kw, 2, pw))
        self.layer2 = self.get_norm_layer(nn.Conv2d(1 * nef, 2 * nef, kw, 2, pw))
        self.layer3 = self.get_norm_layer(nn.Conv2d(2 * nef, 4 * nef, kw, 2, pw))
        self.layer4 = self.get_norm_layer(nn.Conv2d(4 * nef, 8 * nef, kw, 2, pw))
        self.layer5 = self.get_norm_layer(nn.Conv2d(8 * nef, 8 * nef, kw, 2, pw))
        if params.crop_size >=256:
            self.layer6 = self.get_norm_layer(nn.Conv2d(8 * nef, 8 * nef, kw, 2, pw))

        self.s0 = s0 = 4
        self.fc_mu = nn.Linear(nef * 8 * s0 * s0, 256)
        self.fc_var = nn.Linear(nef * 8 * s0 * s0, 256)
        self.params = params

    def get_norm_layer(self, layer):
        if hasattr(layer, 'out_channels'):
            num_ch = getattr(layer, 'out_channels')
        else:
            num_ch = layer.weight.size(0)
        norm_layer = nn.InstanceNorm2d(num_ch, affine=False)
        return nn.Sequential(layer, norm_layer)


    def forward(self, style_image):
        x = style_image
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear')  # 마지막 conv 출력이 (N, 8*nef, s0, s0) 이므로

        x = self.act(self.layer1(x))
        x = self.act(self.layer2(x))
        x = self.act(self.layer3(x))
        x = self.act(self.layer4(x))
        x = self.act(self.layer5(x))
        if self.params.crop_size >= 256:
            x = self.act(self.layer6(x))

        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)

        return mu, logvar