import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from implicit_alignment_v2 import ImplicitWarpModule
    
from _default_essentials import *
from _swinir_3d import SwinIR3d
from _recurrent_backbone import TwoStageBiRecurrentBackbone

@ARCH_REGISTRY.register()
class IARTv2(nn.Module):

    def __init__(self,
                 in_channels=3,
                 embed_dim=32,
                 upsampler_channels=32,
                 out_channels=3,
                 num_stages=2,
                 input_size=(3, 64, 64), 
                 num_heads=4, window_size=(2, 8, 8), shift_size=(1, 2, 2), 
                 mlp_ratio=4.0, mlp_drop=0.1,
                 qkv_bias=True, qk_scale=None, 
                 attn_drop=0.1, fc_drop=0.1, drop_path=0.2, 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
                 spynet_path=None,
                 depths=(4, 4, 4),
                 ):

        super().__init__()

        self.preprocessor       = SpatialPreprocessor2d(in_channels=in_channels, out_channels=embed_dim)
        self.flow_computer      = BidirectionFlowComputer(spynet_path)
        self.upsampler          = SpatialUpsampler4x(in_channels=embed_dim, 
                                                     mid_channels=upsampler_channels, 
                                                     out_channels=out_channels)
        self.flow_warper        = ImplicitWarpModule(dim=embed_dim, image_size=input_size[1:], 
                                                     pe_dim=embed_dim, num_heads=4, pe_temp=0.01)
        self.feature_extractor  = SwinIR3d(block_args=dict(embed_dim=embed_dim, input_size=input_size, 
                                                           num_heads=num_heads, window_size=window_size, 
                                                           shift_size=shift_size, mlp_ratio=mlp_ratio, 
                                                           norm_layer=norm_layer, qkv_bias=qkv_bias, 
                                                           qk_scale=qk_scale, attn_drop=attn_drop, 
                                                           fc_drop=fc_drop, drop_path=drop_path, 
                                                           act_layer=act_layer, mlp_drop=mlp_drop), 
                                           depths=depths)

        self.backbone = TwoStageBiRecurrentBackbone(self.feature_extractor, 
                                                    self.flow_warper,
                                                    num_stages)

    def forward(self, lqs):

        feats = self.preprocessor(lqs)

        flows = self.flow_computer(lqs)

        feats = self.backbone(feats, flows)

        outs = self.upsampler(lqs, feats)

        return outs
    
if __name__ == "__main__":
    # Define model parameters
    in_channels = 3
    embed_dim = 32
    upsampler_channels = 32
    out_channels = 3
    num_stages = 2
    input_size = (3, 64, 64)
    num_heads = 4
    window_size = (2, 8, 8)
    shift_size = (1, 2, 2)
    mlp_ratio = 4.0
    mlp_drop = 0.1
    qkv_bias = True
    qk_scale = None
    attn_drop = 0.1
    fc_drop = 0.1
    drop_path = 0.2
    act_layer = nn.GELU
    norm_layer = nn.LayerNorm
    spynet_path = None
    depths = (4, 4, 4)

    # Initialize the model
    model = IARTv2(in_channels=in_channels,
                 embed_dim=embed_dim,
                 upsampler_channels=upsampler_channels,
                 out_channels=out_channels,
                 num_stages=num_stages,
                 input_size=input_size,
                 num_heads=num_heads,
                 window_size=window_size,
                 shift_size=shift_size,
                 mlp_ratio=mlp_ratio,
                 mlp_drop=mlp_drop,
                 qkv_bias=qkv_bias,
                 qk_scale=qk_scale,
                 attn_drop=attn_drop,
                 fc_drop=fc_drop,
                 drop_path=drop_path,
                 act_layer=act_layer,
                 norm_layer=norm_layer,
                 spynet_path=spynet_path,
                 depths=depths)

    # Create random input data (batch size 1, 3 channels, 64x64 spatial dimensions)
    input_data = torch.randn(1, 3, 3, 64, 64)  # [batch_size, temporal, in_channels, height, width]

    # Perform forward pass
    output = model(input_data)

    # Check the output size
    print(f"Input shape: {input_data.shape}")
    print(f"Output shape: {output.shape}")

