"""
original paper: https://github.com/NVlabs/SPADE
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.archtiecture import SPADEResBlk


class SPADEGenerator(nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")
        return parser

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.init_w, self.init_h = self.get_init_size()
    
        mult = 16
        ngf = params.num_generator_filters
        mask_ch = params.mask_ch
        stride_size = params.spade_stride_size
        kernel_size = params.spade_kernel_size
        pad_size = params.spade_pad_size

        resnet_args = [params.spade_hidden_ch, kernel_size, stride_size, pad_size,
            params.use_spec_norm_g, params.resblk_act, params.use_original_resblk, 
            params.spade_norm, params.spade_act]  # SPADEResBlk arguments

        self.first_layer = nn.Linear(self.params.z_dim, mult * ngf * self.init_w * self.init_h)

        self.head_0 = SPADEResBlk(mult*ngf, mult*ngf, mask_ch, *resnet_args)
        self.middle_0 = SPADEResBlk(mult*ngf, mult*ngf, mask_ch, *resnet_args)
        self.middle_1 = SPADEResBlk(mult*ngf, mult*ngf, mask_ch, *resnet_args)

        up_layers = []
        while mult > 1:
            up_layers.append(SPADEResBlk(mult*ngf, (mult // 2)*ngf, mask_ch, *resnet_args))
            mult = mult // 2

        final_num_ch = ngf
        if self.params.num_upsampling_layers == 'most':
            up_layers.append(SPADEResBlk(1*ngf, ngf // 2, mask_ch, *resnet_args))
            final_num_ch = ngf // 2

        self.up_layers = nn.ModuleList(up_layers)
        self.toRGBImg = nn.Conv(final_num_ch, 3, kernel_size, stride_size, pad_size)
        self.up = nn.Upsample(scale_factor=2, mode=params.upsample_type)

    def get_init_size(self):
        # num upsampling layers: normal=5, more=6, most=7
        if self.params.num_upsampling_layers == 'normal':
            n_up = 5
        elif self.params.num_upsampling_layers == 'more':
            n_up = 6
        elif self.params.num_upsampling_layers == 'most':
            n_up = 7
        else:
            raise ValueError('params.num_upsampling_layers [%s] not recognized' %
                             self.params.num_upsampling_layers)
        init_w = self.params.crop_size // (2**n_up)
        init_h = round(init_w / params.aspect_ratio)
        return init_w, init_h


    def forward(self, mask, z=None):
        if z is None:  # samples random noise z
            z = torch.randn(mask.size(0), self.params.z_dim, 
                    dtype=torch.float32, device=mask.get_device())

        x = self.first_layer(z)
        x = x.view(-1, 16 * self.params.ngf, self.init_w, self.init_h)

        x = self.head_0(x, mask)
        x = self.up(x)

        x = self.middle_0(x, mask)
        if self.params.num_upsampling_layers == 'more' or \
                self.params.num_upsampling_layers == 'most':
            x = self.up(x)
        x = self.middle_1(x, mask)

        for layer in self.up_layers:
            x = self.up(x)
            x = layer(x)
        
        x = self.toRGBImg(x)
        x = F.tanh(x)
        return x