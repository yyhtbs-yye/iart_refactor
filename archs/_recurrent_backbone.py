import torch
import torch.nn as nn

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import flow_warp

@ARCH_REGISTRY.register()
class TwoStageBiRecurrentBackbone(nn.Module):

    def __init__(self,
                 feature_extractor,     # Abstract feature_extractor for feature extraction
                 flow_warper,           # Abstract flow warper (e.g., implicit warping)
                 num_stages):           # Number of recurrent stages

        super().__init__()

        self.feature_extractor       = feature_extractor
        self.flow_warper             = flow_warper
        self.num_stages              = num_stages

    def propagate(self, curr_feats, flows, is_reversed=False):

        n, t, c, h, w = curr_feats.size()
        out_feats = []
        prop_feat = torch.zeros_like(curr_feats[:, 0, ...])

        feat_indices = list(range(t)) if not is_reversed else list(range(t - 1, -1, -1))

        for i in range(0, t):
            
            curr_feat = curr_feats[:, feat_indices[i], ...]
            aligned_n1_feat = curr_feat
            aligned_n2_feat = curr_feat

            if i > 0:
                n1_flow = flows[:, feat_indices[i - 1], ...]

                aligned_n1_feat = self.flow_warper(prop_feat, curr_feat, n1_flow.permute(0, 2, 3, 1))
                
                n2_flow = torch.zeros_like(n1_flow)
                aligned_n2_feat = curr_feat

                if i > 1:
                    n2_flow = flows[:, feat_indices[i - 2], :, :, :]
                    # Compute second-order optical flow using first-order flow.
                    n2_flow = n1_flow + flow_warp(n2_flow, n1_flow.permute(0, 2, 3, 1))
                    n2_feat = out_feats[-2] # The position of 'n-2' to match 'n'
                    aligned_n2_feat = self.flow_warper(n2_feat, curr_feat, n2_flow.permute(0, 2, 3, 1))

            aggr_feat = torch.stack([curr_feat, aligned_n1_feat, aligned_n2_feat], dim=1) # aggr_feat.shape=[B, T, C, H, W]

            prop_feat = self.feature_extractor(aggr_feat) + curr_feat # prop_feat.shape=[B, C, H, W]

            out_feats.append(prop_feat.clone())

        if is_reversed:
            out_feats = out_feats[::-1]

        return torch.stack(out_feats, dim=1)

    def forward(self, feats, flows):

        # Compute forward and backward flows
        forward_flows, backward_flows = flows

        # Propagate features through stages
        for i in range(self.num_stages):
            feats = self.propagate(feats, forward_flows, self.feature_extractor, is_reversed=False)
            feats = self.propagate(feats, backward_flows, self.feature_extractor, is_reversed=True)

        return feats
