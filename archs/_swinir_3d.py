import torch.nn as nn

from _flexible_blocks import FlexibleHomoBlocks
from _swin_transformer import SwinTransformerBlock

from _default_essentials import Conv2dExt

class SwinIR3d(nn.Module):

    def __init__(self, model_args=None, 
                 depths=(4, 4, 4)
                 ):

        super(SwinIR3d, self).__init__()
        if model_args is None:
            model_args = {
                'embed_dim': 64,          # Dimension of the embedding
                'input_size': (3, 64, 64),# Input size for the 3D data (depth, height, width)
                'num_heads': 4,            # Number of attention heads in multi-head attention
                'window_size': (2, 8, 8),  # The window size for the attention operation
                'shift_size': (1, 2, 2),   # Shift size for the windowed attention
                'mlp_ratio': 4.0,          # Ratio for the hidden layer in the MLP (feed-forward) module
                'norm_layer': nn.LayerNorm,# Normalization layer (can be LayerNorm, BatchNorm, etc.)
                'qkv_bias': True,          # Whether to use bias for QKV in attention mechanism
                'qk_scale': None,          # Scaling factor for Q and K in attention (if None, it defaults to √embed_dim)
                'attn_drop': 0.1,          # Dropout rate for attention layer
                'fc_drop': 0.1,            # Dropout rate for the fully connected (MLP) layer
                'drop_path': 0.2,          # Dropout path rate (for stochastic depth)
                'act_layer': nn.GELU,      # Activation function for the MLP layer (GELU in this case)
                'mlp_drop': 0.1            # Dropout rate for the MLP
            }

        self.swin_modules = nn.ModuleList()  # Ensure swin_blocks is a module list if these are modules

        for depth in depths:
            # Construct the blocks step by step for each depth
            swin_module = nn.Sequential(
                Conv2dExt(in_channels=model_args['embed_dim'], 
                        out_channels=model_args['embed_dim'], 
                        kernel_size=3, padding=1),
                
                FlexibleHomoBlocks([SwinTransformerBlock], [model_args], [0] * depth, 
                                prev_block_type='conv2d', 
                                this_block_type='vit', 
                                post_block_type='conv2d', 
                                use_residue=False),
                
                Conv2dExt(in_channels=model_args['embed_dim'], 
                        out_channels=model_args['embed_dim'], 
                        kernel_size=3, padding=1)
            )
            
            # Append the constructed blocks for each depth to the module list
            self.swin_modules.append(swin_module)

        
    def forward(self, x):
        
        for swin_module in self.swin_modules:
            x = swin_module(x) + x

        return x