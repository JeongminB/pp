"""
original paper: https://github.com/NVlabs/SPADE
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork


class MultiscaleDiscriminator(BaseNetwork):
    def __init__(self, params):
        super().__init__()
        self.params = params
        
        for i in range(params.num_D):
            subD = self.create_single_discriminator(params)
            self.add_module('discriminator_%d' % i, subD)  # adds a child module (discriminator)

    def create_single_discriminator(self, params):
        netD = PatchDiscimiantor(params)
        return netD

    def forward(self, x):
        pass


class PatchDiscimiantor(BaseNetwork):
    def __init__(self):
        super().__init__()

    
    def forward(self, x):
        pass