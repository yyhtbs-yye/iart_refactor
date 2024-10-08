import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import lru_cache
import math

from ._utils import window_partition, window_reverse, compute_padding
from ._building_block import SelfAttention3D, Mlp
from ._dropout import DropPath

class SwinTransformerBlock(nn.Module):

    def __init__(self, dim, data_size, vit_args):
        
        super(SwinTransformerBlock, self).__init__()

        self.dim            = dim
        self.data_size      = data_size
        self.num_heads      = vit_args['num_heads']
        self.window_size    = vit_args['window_size']
        self.shift_size     = vit_args['shift_size']
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

    def forward(self, x):
        b, t, h, w, c = x.shape

        padding   = compute_padding(self.window_size, t, h, w)

        attn_mask = compute_mask((t, h, w), tuple(self.window_size), self.shift_size,).to(x.device)

        x_copy = x

        x = self.pre_norm(x)

        # pad feature maps to multiples of window size
        x = F.pad(x, (0, 0, *padding))

        _, pT, pH, pW, _ = x.shape

        # cyclic shift
        x = torch.roll(
            x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))

        # partition windows, output shape = [nWin*b, wT*wH*wW, c]
        x_windows = window_partition(x, self.window_size)  

        attn_windows = self.attn(x_windows, mask=attn_mask, nW=x_windows.size(0)//x.size(0))

        # reshape attended, output shape = [nWin*b, wT, wH, wW, c]
        attn_windows = attn_windows.view(-1, *self.window_size, c)

        # merge attended, output shape = [b, pT, pH, pW, c]
        x = window_reverse(attn_windows, self.window_size, b, pT, pH, pW) 
        
        # reverse cyclic shift
        x = torch.roll(x, 
                       shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), 
                       dims=(1, 2, 3))

        x = x[:, :t, :h, :w, :] #.contiguous()

        # FFN, Residue 
        x = self.drop_path(x) + x_copy
        x = x + self.drop_path(self.mlp(self.epi_norm(x)))

        return x
    

@lru_cache()
def compute_mask(data_size, window_size, shift_size):
    t, h, w = data_size
    pT = int(math.ceil(t / window_size[0])) * window_size[0]
    pH = int(math.ceil(h / window_size[1])) * window_size[1]
    pW = int(math.ceil(w / window_size[2])) * window_size[2]
    
    img_mask = torch.zeros((1, pT, pH, pW, 1))  # 1 D H W 1
    
    cnt = 0
    for d in [slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None)]:
        for h in [slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None)]:
            for w in [slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None)]:
                img_mask[:, d, h, w, :] = cnt
                cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.view(-1, window_size[0] * window_size[1] * window_size[2])
    
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    
    return attn_mask

