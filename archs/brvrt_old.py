import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.archs.arch_util import flow_warp
from basicsr.archs.spynet_arch import SpyNet
from basicsr.utils.registry import ARCH_REGISTRY
from .backbone_arch import VallinaBackbone
from .swin_transformer import SwinTransformerBlock
from .implicit_alignment_v2 import ImplicitWarpModule
    
# Name: Bidirectional Recurrent Video Restoration Transformer

@ARCH_REGISTRY.register()
class BRVRT(nn.Module):

    def __init__(self,
                 in_channels=3,
                 mid_channels=32,
                 embed_dim=64, 
                 img_size=64,
                 num_stages=2,
                 spynet_path=None,
                 preprocessor_name=None, 
                 preprocessor_args=None,
                 flow_warper_name=None, 
                 flow_warper_args=None, 
                 backbone_name=None, 
                 backbone_args=None,
                 construction_name=None, 
                 construction_args=None,
                 ):

        super().__init__()
        self.mid_channels = mid_channels
        self.embed_dim = embed_dim
        self.conv_before_upsample = nn.Conv2d(embed_dim, mid_channels, 3, 1, 1)

        self.img_size = img_size

        if not isinstance(img_size, list):
            self.img_size = [img_size, img_size]

        # optical flow
        self.spynet = SpyNet(spynet_path)
        self.num_stages = num_stages

        self.spatial_conv = nn.Conv2d(in_channels, self.embed_dim, 3, 1, 1)

        self.num_frames = num_frames

        if rvtm_pre_name is None:
            rvtm_pre_name = Conv2dFor5d  # For instance, a simple 3D Conv layer can be used here as a placeholder
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
            rvtm_epi_name = Conv2dFor5d  # Again, a placeholder for the epi layer
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

        self.upconv1 = nn.Conv2d(mid_channels, mid_channels * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(mid_channels, 64 * 4, 3, 1, 1, bias=True)

        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # check if the sequence is augmented by flipping
        self.is_mirror_extended = False

        self.implicit_warp = ImplicitWarpModule(
            dim=embed_dim, image_size=self.img_size,
            pe_dim=embed_dim,
            num_heads=4,
            pe_temp=0.01)
        
        self.feat_indices_fwd = None # 
        self.feat_indices_bwd = None # list(range(-1, -num_frames - 1, -1))

    def compute_flow(self, lqs):

        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)

        backward_flows = self.spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)

        if self.is_mirror_extended:  # forward_flows = backward_flows.flip(1)
            forward_flows = None
        else:
            forward_flows = self.spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)

        return forward_flows, backward_flows

    def propagate(self, curr_feats, flows, fextor, is_reversed=False):

        n, t, c, h, w = curr_feats.size()
        # t = self.num_frames

        out_feats = list()
        prop_feat = curr_feats.new_zeros(n, self.mid_channels, h, w)
        # align_feat = curr_feats.new_zeros(n, self.mid_channels, h, w)
        # n1_feat = prop_feat

        self.feat_indices_fwd = list(range(t))
        self.feat_indices_bwd = list(range(-1, -t - 1, -1))

        feat_indices = self.feat_indices_fwd if is_reversed else self.feat_indices_bwd

        for i in range(0, t):
            
            curr_feat = curr_feats[:, feat_indices[i], ...]
            n1_cond = curr_feat
            n2_cond = curr_feat

            if i > 0:
                n1_flow = flows[:, feat_indices[i - 1], ...]

                n1_cond = self.implicit_warp(prop_feat, curr_feat, n1_flow.permute(0, 2, 3, 1))
                
                n2_feat = curr_feat
                n2_flow = torch.zeros_like(n1_flow)
                n2_cond = curr_feat
                if i > 1:
                    n2_flow = flows[:, feat_indices[i - 2], :, :, :]
                    # Compute second-order optical flow using first-order flow.
                    n2_flow = n1_flow + flow_warp(n2_flow, n1_flow.permute(0, 2, 3, 1))
                    n2_feat = out_feats[-2] # The position of 'n-2' to match 'n'
                    n2_cond = self.implicit_warp(n2_feat, curr_feat, n2_flow.permute(0, 2, 3, 1))

                # Concatenate conditions for deformable convolution.
            n12c_cond = torch.stack([n1_cond, curr_feat, n2_cond], dim=1)

            prop_feat = fextor(n12c_cond)[:, self.num_frames // 2, :, :, :] + curr_feat

            out_feats.append(prop_feat.clone())

        if is_reversed:
            out_feats = out_feats[::-1]

        return torch.stack(out_feats, dim=1)

    def upsample(self, lqs, feats):

        N, T, C, H, W = feats.shape

        outputs = []

        for t in range(0, T):
            h = feats[:, t, ...]
            h = self.conv_before_upsample(h)
            h = self.lrelu(self.pixel_shuffle(self.upconv1(h)))
            h = self.lrelu(self.pixel_shuffle(self.upconv2(h)))
            h = self.lrelu(self.conv_hr(h))
            y = self.conv_last(h)
            y += self.img_upsample(lqs[:, t, :, :, :])

            outputs.append(y)

        return torch.stack(outputs, dim=1)

    def forward(self, lqs):

        n, t, c, h, w = lqs.size()

        feats = self.spatial_conv(lqs.view(-1, c, h, w)).view(n, t, -1, h, w)

        forward_flows, backward_flows = self.compute_flow(lqs)

        # feature propgation
        for i in range(self.num_stages):

            feats = self.propagate(feats, forward_flows, self.fextors[2*i+0], is_reversed=False)
            feats = self.propagate(feats, backward_flows, self.fextors[2*i+1], is_reversed=True)

        return self.upsample(lqs, feats)


class Conv2dFor5d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dFor5d, self).__init__()
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