import torch
import torch.nn as nn
from basicsr.archs.spynet_arch import SpyNet

class SpatialUpsampler4x(nn.Module):
    def __init__(self, in_channels, mid_channels=64, out_channels=3):
        super(SpatialUpsampler4x, self).__init__()
        
        # Define the layers used in the upsampling process
        self.conv_before_upsample = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.upconv1 = nn.Conv2d(mid_channels, mid_channels * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(mid_channels, mid_channels * 4, 3, 1, 1, bias=True)
        
        # Pixel shuffle for upscaling
        self.pixel_shuffle = nn.PixelShuffle(2)
        
        # Convolution layers for high-resolution features
        self.conv_hr = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.conv_last = nn.Conv2d(mid_channels, out_channels, 3, 1, 1)
        
        # Upsample layer for final resolution increase
        self.img_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        
        # Activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, lqs, feats):

        N, T, C, H, W = feats.shape
        outputs = []

        # Loop over each frame in the sequence
        for t in range(T):
            h = feats[:, t, ...]
            
            # Apply the series of convolutions and pixel shuffle for upscaling
            h = self.conv_before_upsample(h)
            h = self.lrelu(self.pixel_shuffle(self.upconv1(h)))
            h = self.lrelu(self.pixel_shuffle(self.upconv2(h)))
            h = self.lrelu(self.conv_hr(h))
            y = self.conv_last(h)
            
            # Add the upsampled low-quality input to the output
            y += self.img_upsample(lqs[:, t, :, :, :])
            outputs.append(y)

        # Stack the outputs across the time dimension
        return torch.stack(outputs, dim=1)

class SpatialPreprocessor2d(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1):
        super(SpatialPreprocessor2d, self).__init__()
        
        self.spatial_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, lqs):

        b, t, c, h, w = lqs.shape
        
        feats = self.spatial_conv(lqs.view(-1, c, h, w)).view(b, t, -1, h, w)

        return feats

class BidirectionFlowComputer(nn.Module):

    def __init__(self, spynet_path):
        super(BidirectionFlowComputer, self).__init__()

        self.spynet = SpyNet(spynet_path)

    def forward(self, lqs):

        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)

        backward_flows = self.spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)

        forward_flows  = self.spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)

        return forward_flows, backward_flows


class Conv2dExt(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dExt, self).__init__()
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