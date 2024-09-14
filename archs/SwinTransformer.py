import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import lru_cache
import math

from ._utils import window_partition, window_reverse
from ._building_block import SelfAttention3D, Mlp
from ._dropout import DropPath

class SwinTransformerBlock(nn.Module):

    def __init__(self, dim, data_size, vit_args):
        
        super(SwinTransformerBlock).__init__()

        self.dim            = dim
        self.data_size      = data_size
        self.num_heads      = vit_args['num_heads']
        self.window_size    = vit_args['window_size']
        self.shift_size     = vit_args['shift_size']
        self.mlp_ratio      = vit_args['mlp_ratio']
        self.pre_norm       = vit_args['norm_layer'](dim)
        self.post_norm      = vit_args['norm_layer'](dim)

        self.register_buffer("attn_mask", compute_mask(data_size,
                                                       tuple(self.window_size),
                                                       self.shift_size,))


        self.attn = SelfAttention3D(dim=dim, 
                                    window_size=self.window_size, num_heads=vit_args['num_heads'],
                                    qkv_bias=vit_args['qkv_bias'], qk_scale=vit_args['qk_scale'],
                                    attn_drop=vit_args['attn_drop'], fc_drop=vit_args['fc_drop'])

        self.drop_path = DropPath(vit_args['drop_path']) if vit_args['drop_path'] > 0. else nn.Identity()

        self.mlp = Mlp(in_features=dim, mid_features=dim * vit_args['mlp_ratio'], 
                       act_layer=vit_args['act_layer'], drop=vit_args['mlp_drop'])

    def forward(self, x):
        b, t, h, w, c = x.shape

        x_copy = x

        x = self.pre_norm(x)

        # pad feature maps to multiples of window size
        pad_w0 = pad_h0 = pad_t0 = 0
        pad_t1 = (self.window_size[0] - t % self.window_size[0]) % self.window_size[0]
        pad_h1 = (self.window_size[1] - h % self.window_size[1]) % self.window_size[1]
        pad_w1 = (self.window_size[2] - w % self.window_size[2]) % self.window_size[2]
        x = F.pad(x, (0, 0, pad_w0, pad_w1, pad_h0, pad_h1, pad_t0, pad_t1))

        _, Tp, Hp, Wp, _ = x.shape

        # cyclic shift
        x = torch.roll(
            x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))

        # partition windows, output shape = [nWin*b, wT*wH*wW, c]
        x_windows = window_partition(x, self.window_size)  

        # W-MSA/SW-MSA 
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # reshape attended, output shape = [nWin*b, wT, wH, wW, c]
        attn_windows = attn_windows.view(-1, *(self.window_size + [ c ]))
        # merge attended, output shape = [b, Tp, Hp, Wp, c]
        x = window_reverse(attn_windows, self.window_size, b, Tp, Hp, Wp) 
        
        # reverse cyclic shift
        x = torch.roll(x, 
                       shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), 
                       dims=(1, 2, 3))

        if pad_t1 > 0 or pad_w1 > 0 or pad_h1 > 0:
            x = x[:, :t, :h, :w, :].contiguous()

        # FFN, Residue
        x = self.drop_path(x) + x_copy
        x = x + self.drop_path(self.mlp(self.post_norm(x)))

        return x
    

@lru_cache()
def compute_mask(data_size, window_size, shift_size):
    t, h, w = data_size
    Tp = int(math.ceil(t / window_size[0])) * window_size[0]
    Hp = int(math.ceil(h / window_size[1])) * window_size[1]
    Wp = int(math.ceil(w / window_size[2])) * window_size[2]
    
    img_mask = torch.zeros((1, Tp, Hp, Wp, 1))  # 1 D H W 1
    
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

