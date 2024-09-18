import torch
import torch.nn as nn
# from torchinfo import summary  # Optional for model summary, install via `pip install torchinfo`

# Assuming the provided code is saved in a file named `video_transformer.py`
from archs.VideoTransformer import VideoTransformerBackbone
from archs.SwinTransformer import SwinTransformerBlock

def test_video_transformer():
    # Random input tensor [batch_size, time_steps, channels, height, width]
    batch_size = 2
    time_steps = 4
    channels = 3
    height = 224
    width = 224
    input_tensor = torch.randn(batch_size, time_steps, channels, height, width)

    # Define some placeholder arguments for the model (these need to be valid for your model)
    dim = 8
    data_size = (time_steps, height, width)

    # Pre, Vit, and Epi layer arguments (dummy placeholders, adjust as needed)
    rvtm_pre_name = nn.Conv3d  # For instance, a simple 3D Conv layer can be used here as a placeholder
    rvtm_pre_args = {'in_channels': channels, 'out_channels': dim, 'kernel_size': 3, 'padding': 1}
    
    vit_namex = [SwinTransformerBlock] * 2  # Dummy placeholder for Vit layers
    vit_argx = [{'num_heads': 4, 'window_size': (2, 3, 3), 'shift_size': (1, 1, 1), 'mlp_ratio': 4, 'norm_layer': nn.LayerNorm, 
                 'act_layer': nn.LeakyReLU, 
                 'qkv_bias': True, 'qk_scale': None, 'attn_drop': 0.1, 'fc_drop': 0.1, 'mlp_drop': 0.1, 'drop_path': 0.1}] * 2
    vit_seqx = [[0], [1]]  # Sequence indices for layers
    
    rvtm_epi_name = nn.Conv2d  # Again, a placeholder for the epi layer
    rvtm_epi_args = {'in_channels': dim, 'out_channels': dim, 'kernel_size': 3, 'padding': 1}
    
    # Create model
    model = VideoTransformerBackbone(dim=dim, data_size=data_size,
                                     rvtm_pre_name=rvtm_pre_name, rvtm_pre_args=rvtm_pre_args,
                                     vit_namex=vit_namex, vit_argx=vit_argx, vit_seqx=vit_seqx,
                                     rvtm_epi_name=rvtm_epi_name, rvtm_epi_args=rvtm_epi_args)
    
    # Optionally, print a model summary
    # print(summary(model, input_size=(batch_size, time_steps, channels, height, width)))

    # Run forward pass
    output = model(input_tensor)
    print("Output shape:", output.shape)

if __name__ == "__main__":
    test_video_transformer()
