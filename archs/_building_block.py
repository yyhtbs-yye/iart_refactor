import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from ._positional_encoding import get_relative_position_index_3d

class Mlp(nn.Module):

    def __init__(self, in_features, 
                 mid_features=None, 
                 out_features=None, 
                 act_layer=nn.GELU, 
                 drop=0.):

        super().__init__()
        out_features = out_features or in_features
        mid_features = mid_features or in_features
        self.fc1 = nn.Linear(in_features, mid_features)
        self.fc2 = nn.Linear(mid_features, out_features)
        self.drop = nn.Dropout(drop)
        self.act = act_layer()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class SelfAttention3D(nn.Module):

    def __init__(self, dim, token_dim, num_heads, 
                 qkv_bias=True, qk_scale=None, 
                 attn_drop=0., fc_drop=0.):

        super().__init__()
        self.dim = dim
        self.token_dim = token_dim  # Wh, Ww
        self.num_heads = num_heads  
        head_dim = dim // num_heads  
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.rpt = nn.Parameter(
            torch.zeros((2 * token_dim[0] - 1) * (2 * token_dim[1] - 1) * (2 * token_dim[2] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        self.register_buffer("rpi", get_relative_position_index_3d(self.token_dim))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.fc = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.fc_drop = nn.Dropout(fc_drop)

        trunc_normal_(self.rpt, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None, nW=1):
        
        B_, N, D = x.shape

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4).contiguous() # 3, B_, nH, N, D
        q, k, v = qkv[0], qkv[1], qkv[2]                                # each: B_, nH, N, D

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        rpe = self.rpt[self.rpi[:N, :N].reshape(-1)].reshape(N, N, -1)   # Wd*Wh*Ww,Wd*Wh*Ww,nH
        attn = attn + rpe.permute(2, 0, 1).contiguous().unsqueeze(0)    # B_, nH, N, N

        if mask is not None:
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, D)
        x = self.fc(x)
        x = self.fc_drop(x)
        return x
