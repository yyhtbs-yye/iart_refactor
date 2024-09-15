import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import lru_cache
import math

from ._utils import window_partition, window_reverse, compute_padding
from ._building_block import SelfAttention3D, Mlp
from ._dropout import DropPath

class WinTransformerBlock(nn.Module):

    def __init__(self, dim, data_size, vit_args):
        
        super(WinTransformerBlock, self).__init__()

        self.dim            = dim
        self.data_size      = data_size
        self.num_heads      = vit_args['num_heads']
        self.window_size    = vit_args['window_size']
        self.mlp_ratio      = vit_args['mlp_ratio']
        self.pre_norm       = vit_args['norm_layer'](dim)
        self.epi_norm       = vit_args['norm_layer'](dim)

        self.attn           = SelfAttention3D(dim=dim, 
                                              token_dim=self.window_size, num_heads=vit_args['num_heads'],
                                              qkv_bias=vit_args['qkv_bias'], qk_scale=vit_args['qk_scale'],
                                              attn_drop=vit_args['attn_drop'], fc_drop=vit_args['fc_drop'])

        self.drop_path      = DropPath(vit_args['drop_path']) if vit_args['drop_path'] > 0. else nn.Identity()

        self.mlp            = Mlp(in_features=dim, mid_features=dim * vit_args['mlp_ratio'], 
                                  act_layer=vit_args['act_layer'], drop=vit_args['mlp_drop'])

        self.padding        = compute_padding(self.window_size, *data_size)

    def forward(self, x):
        b, t, h, w, c = x.shape

        x_copy = x

        x = self.pre_norm(x)

        # pad feature maps to multiples of window size
        x = F.pad(x, (0, 0, *self.padding))

        _, pT, pH, pW, _ = x.shape

        # partition windows, output shape = [nWin*b, wT*wH*wW, c]
        x_windows = window_partition(x, self.window_size)  

        attn_windows = self.attn(x_windows)

        # reshape attended, output shape = [nWin*b, wT, wH, wW, c]
        attn_windows = attn_windows.view(-1, *(self.window_size + [ c ]))

        # merge attended windows, output shape = [b, pT, pH, pW, c]
        x = window_reverse(attn_windows, self.window_size, b, pT, pH, pW) 
        
        # remove padding
        x = x[:, :t, :h, :w, :] 

        # FFN and residual connection
        x = self.drop_path(x) + x_copy
        x = x + self.drop_path(self.mlp(self.epi_norm(x)))

        return x
    