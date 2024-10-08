import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.archs.arch_util import flow_warp
from basicsr.archs.spynet_arch import SpyNet
from basicsr.utils.registry import ARCH_REGISTRY
from .backbone_arch import VallinaBackbone
from .swin_transformer import SwinTransformerBlock
from .implicit_alignment_v2 import ImplicitWarpModule
    
from _default_essentials import *

@ARCH_REGISTRY.register()
class IART(nn.Module):

    def __init__(self,
                 in_channels=3,
                 embed_dim=32,
                 upsampler_channels=32,
                 out_channels=3,
                 img_size=64,
                 num_stages=2,
                 spynet_path=None,
                 ):

        super().__init__()

        self.preprocessor   = SpatialPreprocessor2d(in_channels=in_channels, out_channels=embed_dim)
        self.flow_computer  = BidirectionFlowComputer(spynet_path)
        self.upsampler      = SpatialUpsampler4x(in_channels=embed_dim, 
                                                 mid_channels=upsampler_channels, 
                                                 out_channels=out_channels)
        self.flow_warper    = ImplicitWarpModule(dim=embed_dim, image_size=img_size, 
                                                 pe_dim=embed_dim, num_heads=4, pe_temp=0.01)




        if rvtm_pre_name is None:
            rvtm_pre_name = Conv2dExt  # For instance, a simple 3D Conv layer can be used here as a placeholder
        if rvtm_pre_args is None:
            rvtm_pre_args = {'in_channels': self.embed_dim, 'out_channels': self.embed_dim, 
                            'kernel_size': 3, 'padding': 1}
        if vit_namex is None:
            vit_namex = [SwinTransformerBlock, SwinTransformerBlock]  # Dummy placeholder for Vit layers
        
        if vit_argx is None:
            vit_argx = [{'num_heads': 4, 'window_size': (2, 8, 8), 'shift_size': (1, 2, 2), 'mlp_ratio': 4, 
                        'norm_layer': nn.LayerNorm, 'act_layer': nn.LeakyReLU, 
                        'qkv_bias': True, 'qk_scale': None, 'attn_drop': 0.1, 'fc_drop': 0.1, 'mlp_drop': 0.1, 'drop_path': 0.1}] * 2
        if vit_seqx is None:
            vit_seqx = [[0, 1], [0, 1]]  # Sequence indices for layers
        
        if rvtm_epi_name is None:
            rvtm_epi_name = Conv2dExt  # Again, a placeholder for the epi layer
        if rvtm_epi_args is None:
            rvtm_epi_args = {'in_channels': self.embed_dim, 'out_channels': self.embed_dim, 'kernel_size': 3, 'padding': 1}

        # propagation branches
        self.fextors = nn.ModuleList([
            VallinaBackbone(dim=self.embed_dim, data_size=(self.num_frames, *self.img_size),
                            rvtm_pre_name=rvtm_pre_name, rvtm_pre_args=rvtm_pre_args,
                            vit_namex=vit_namex, vit_argx=vit_argx, vit_seqx=vit_seqx,
                            rvtm_epi_name=rvtm_epi_name, rvtm_epi_args=rvtm_epi_args)

            for i in range(num_stages*2)
        ])

        self.flow_warper = ImplicitWarpModule(dim=embed_dim, 
                                              image_size=self.img_size,
                                              pe_dim=embed_dim,
                                              num_heads=4,
                                              pe_temp=0.01)
        

    def forward(self, lqs):

        n, t, c, h, w = lqs.size()

        feats = self.spatial_conv(lqs.view(-1, c, h, w)).view(n, t, -1, h, w)

        forward_flows, backward_flows = self.compute_flow(lqs)

        # feature propgation
        for i in range(self.num_stages):

            feats = self.propagate(feats, forward_flows, self.fextors[2*i+0], is_reversed=False)
            feats = self.propagate(feats, backward_flows, self.fextors[2*i+1], is_reversed=True)

        return self.upsample(lqs, feats)