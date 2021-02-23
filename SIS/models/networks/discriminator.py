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
import torch.nn.utils.spectral_norm as spectral_norm


class MultiscaleDiscriminator(BaseNetwork):
    def __init__(self, params):
        super().__init__()
        self.params = params
        
        for i in range(params.num_D):
            subD = self.create_single_discriminator(self.params)
            self.add_module('discriminator_%d' % i, subD)  # adds a child module (discriminator)

    def create_single_discriminator(self, params):
        netD = PatchDiscimiantor(params)
        return netD

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3,
                            stride=2, padding=[1, 1],
                            count_include_pad=False)  # count_include_pad: include the zero-padding


    def forward(self, input):
        result = []
        get_intermediate_features = not self.opt.no_gan_fm_loss
        for _, D in self.named_children():  # (name, child)
            out = D(input)
            if self.opt.no_gan_fm_loss:
                out = [out]  # only need the last output of the D
            result.append(out)
            input = self.downsample(input)

        return result


class PatchDiscimiantor(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--num_layers_d', type=int, default=4,
                            help='the number of layers in discriminator')
        return parser
    
    def __init__(self, params):
        super().__init__()
        self.params = params

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        ndf = params.num_discriminator_filters
        input_cn = self.get_init_size()

        sequence = [[nn.Conv2d(input_cn, ndf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, False)]]

        for n in range(1, params.num_layers_d):
            nf_prev = ndf
            nf = min(nf * 2, 512)
            stride = 1 if n == params.num_layers_d - 1 else 2
            sequence += [[self.norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=stride, padding=padw)),
                          nn.LeakyReLU(0.2, False)]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def get_init_size(self):
        input_ch = self.params.mask_ch + self.params.output_ch
        if self.params.contain_dontcare_label:
            input_ch += 1
        if not self.params.no_instance:
            input_ch += 1
        return input_ch       

    def norm_layer(self, layer):
        if self.params.use_spec_norm_d:
            layer = spectral_norm(layer)
        num_ch = getattr(layer, 'out_channels')
        norm_layer = nn.InstanceNorm2d(num_ch, affine=False)
        return nn.Sequential(layer, norm_layer)


    def forward(self, x):
        results = [x]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        get_intermediate_features = not self.params.no_gan_fm_loss
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]