import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from models.normalization import SPADE


class SPADEResBlk(nn.Module):
    def __init__(self, in_ch, out_ch, mask_ch, 
            hidden_ch=128, kernel_size=3, stride_size=1, pad_size=1,
            use_spec_norm=True, act='relu', use_original_resblk=True,
            spade_norm='batch', spade_act='relu'):
        super().__init__()
        
        self.act = act                          # the activation function in the SPADE Residual Block
        self.learn_shortcut = (in_ch!= out_ch)  # (SPADE-ACT-CONV) unit used when the input size does not match the output size
        mid_ch = min(in_ch, out_ch)             

        self.conv_0 = nn.Conv2d(in_ch, mid_ch, kernel_size, stride_size, pad_size)
        self.conv_1 = nn.Conv2d(mid_ch, out_ch, kernel_size, stride_size, pad_size)
        if self.learn_shortcut:
            self.conv_shortcut = nn.Conv2d(in_ch, out_ch, kernel_size, stride_size, pad_size, bias=False)
        
        if use_spec_norm:
           self.conv_0 = spectral_norm(self.conv_0) 
           self.conv_1 = spectral_norm(self.conv_1)
           if self.learn_shortcut:
               self.conv_shortcut = spectral_norm(self.conv_shortcut)

        args = [hidden_ch, kernel_size, stride_size, pad_size, spade_norm, spade_act]  # SPADE arguments
        self.spade_0 = SPADE(in_ch, mask_ch, *args)
        self.spade_1 = SPADE(mid_ch, mask_ch, *args)
        if self.learn_shortcut:
            self.spade_shortcut = SPADE(in_ch, mask_ch, *args)
        
        self.use_original_resblk = use_original_resblk

    def actv(self, x):
        if self.act == 'relu':
            return F.relu(x, inplace=True)
        elif self.act == 'leaky_relu':
            return F.leaky_relu(x, 0.2, inplace=True)


    def forward(self, x, mask):
        # architecture:
        #   ┌─────────(SPADE-ReLU-CONV)────────┐ 
        # x─┴─SPADE-ReLU-CONV──SPADE-ReLU-CONV─┴─output
        if self.learn_shortcut:
            if self.use_original_resblk:
                x_shortcut = self.conv_shortcut(self.spade_shortcut(x, mask))  # same as the official code
            else:
                x_shortcut = self.conv_shortcut(self.actv(self.spade_shortcut(x, mask)))
        else:
            x_shortcut = x
        
        x_res = self.conv_0(self.actv(self.spade_0(x, mask)))
        x_res = self.conv_1(self.actv(self.spade_1(x_res, mask)))
        output = x_res + x_shortcut
        return output
    
