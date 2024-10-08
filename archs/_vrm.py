import torch.nn as nn
from basicsr.utils.registry import ARCH_REGISTRY
    
@ARCH_REGISTRY.register()
class VRM(nn.Module):

    def __init__(self,
                 preprocessor,          # Abstract preprocessor (e.g., convolutions)
                 flow_computer,         # Abstract flow computer (e.g., SPyNet)
                 backbone,              # Abstract backbone for feature extraction
                 upsampler,             # Abstract upsampler
                 ):           

        super().__init__()
        self.preprocessor   = preprocessor,          
        self.flow_computer  = flow_computer,         
        self.backbone       = backbone,              
        self.upsampler      = upsampler,             

    def forward(self, lqs):

        feats = self.preprocessor(lqs)

        flows = self.flow_computer(lqs)

        feats = self.backbone(feats, flows)

        outs = self.upsampler(lqs, feats)

        return outs
