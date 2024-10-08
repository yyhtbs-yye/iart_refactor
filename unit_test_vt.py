import torch
import torch.nn as nn
# from torchinfo import summary  # Optional for model summary, install via `pip install torchinfo`

# Assuming the provided code is saved in a file named `video_transformer.py`
from archs.backbone_arch import VallinaBackbone
from archs.swin_transformer import SwinTransformerBlock

class Conv2dFor5D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dFor5D, self).__init__()
        # Use nn.Conv2d with the same parameters
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        # Input shape: [batch_size, time_steps, channels, height, width]
        b, t, c, h, w = x.shape

        # Reshape input to [batch_size * time_steps, channels, height, width]
        x = x.view(b * t, c, h, w)

        # Apply Conv2d
        x = self.conv2d(x)

        # Get the new height and width from the output
        _, c_out, h_out, w_out = x.shape

        # Reshape back to [batch_size, time_steps, out_channels, height, width]
        x = x.view(b, t, c_out, h_out, w_out)

        return x

def test_video_transformer():
    # Random input tensor [batch_size, time_steps, channels, height, width]
    batch_size = 2
    time_steps = 4
    channels = 8
    height = 223
    width = 222
    input_tensor = torch.randn(batch_size, time_steps, channels, height, width)

    # Define some placeholder arguments for the model (these need to be valid for your model)
    dim = 8
    data_size = (time_steps, height, width)

    # Pre, Vit, and Epi layer arguments (dummy placeholders, adjust as needed)
    rvtm_pre_name = Conv2dFor5D  # For instance, a simple 3D Conv layer can be used here as a placeholder
    rvtm_pre_args = {'in_channels': channels, 'out_channels': dim, 'kernel_size': 3, 'padding': 1}
    
    vit_namex = [SwinTransformerBlock] * 2  # Dummy placeholder for Vit layers
    vit_argx = [{'num_heads': 4, 'window_size': (2, 8, 8), 'shift_size': (1, 2, 2), 'mlp_ratio': 4, 
                 'norm_layer': nn.LayerNorm, 'act_layer': nn.LeakyReLU, 
                 'qkv_bias': True, 'qk_scale': None, 'attn_drop': 0.1, 'fc_drop': 0.1, 'mlp_drop': 0.1, 'drop_path': 0.1}] * 2
    vit_seqx = [[0], [1]]  # Sequence indices for layers
    
    rvtm_epi_name = Conv2dFor5D  # Again, a placeholder for the epi layer
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

