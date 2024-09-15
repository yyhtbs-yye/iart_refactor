import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import lru_cache
import math

from ._utils import patch_partition, patch_reverse, compute_padding
from ._building_block import SelfAttention3D, Mlp
from ._dropout import DropPath

class PatchTransformerBlock(nn.Module):

    def __init__(self, dim, data_size, vit_args):
        
        super(PatchTransformerBlock, self).__init__()

        self.dim            = dim
        self.data_size      = data_size
        self.num_heads      = vit_args['num_heads']
        self.patch_size     = vit_args['patch_size']
        self.mlp_ratio      = vit_args['mlp_ratio']
        self.pre_norm       = vit_args['norm_layer'](dim)
        self.epi_norm       = vit_args['norm_layer'](dim)
        self.token_dim      = [(it + jt - 1) // jt for it, jt in zip(data_size, self.patch_size)]

        self.attn           = SelfAttention3D(dim=dim, 
                                              token_dim=self.token_dim, num_heads=vit_args['num_heads'],
                                              qkv_bias=vit_args['qkv_bias'], qk_scale=vit_args['qk_scale'],
                                              attn_drop=vit_args['attn_drop'], fc_drop=vit_args['fc_drop'])

        self.drop_path      = DropPath(vit_args['drop_path']) if vit_args['drop_path'] > 0. else nn.Identity()

        self.mlp            = Mlp(in_features=dim, mid_features=dim * vit_args['mlp_ratio'], 
                                  act_layer=vit_args['act_layer'], drop=vit_args['mlp_drop'])

        self.padding        = compute_padding(self.patch_size, *data_size)

    def forward(self, x):
        b, t, h, w, c = x.shape

        x_copy = x

        x = self.pre_norm(x)

        # pad feature maps to multiples of window size
        x = F.pad(x, (0, 0, *self.padding))

        _, pT, pH, pW, _ = x.shape

        # partition windows, output shape = [b, nPatch, c*pT*pH*pW]
        x_patches = patch_partition(x, self.patch_size)  

        attn_patches = self.attn(x_patches)

        # reshape attended, output shape = [nWin*b, wT, wH, wW, c]
        attn_patches = attn_patches.view(-1, *(self.window_size + [ c ]))

        # merge attended windows, output shape = [b, pT, pH, pW, c]
        x = patch_reverse(attn_patches, self.patch_size, [b, pT, pH, pW, c]) 
        
        # remove padding
        x = x[:, :t, :h, :w, :] 

        # FFN and residual connection
        x = self.drop_path(x) + x_copy
        x = x + self.drop_path(self.mlp(self.epi_norm(x)))

        return x
    