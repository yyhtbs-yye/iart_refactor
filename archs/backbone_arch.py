import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

class BasicLayer(nn.Module):

    def __init__(self, dim, data_size, 
                 vit_namex, vit_argx, vit_seqs,
                 ):

        super().__init__()

        # build blocks
        self.blocks = nn.ModuleList([
            vit_namex[bid](dim=dim, data_size=data_size, 
                           vit_args=vit_argx[bid]) 
                for bid in vit_seqs
        ])

    def forward(self, x):
        # 
        x = x.permute(0, 1, 3, 4, 2)

        for blk in self.blocks:
            x = blk(x)

        x = x.permute(0, 1, 4, 2, 3)

        return x

class RVTM(nn.Module): # Residue Video Transformer Module

    def __init__(self, dim, data_size, 
                 rvtm_pre_name, rvtm_pre_args,
                 vit_namex, vit_argx, vit_seqs,
                 rvtm_epi_name, rvtm_epi_args,
                 ):
        
        super(RVTM, self).__init__()

        self.pre_layer = rvtm_pre_name(**rvtm_pre_args)

        self.vit_layer = BasicLayer(dim, data_size, 
                                    vit_namex, vit_argx, vit_seqs)

        self.epi_layer = rvtm_epi_name(**rvtm_epi_args)

    def forward(self, x):
        return self.epi_layer(self.vit_layer(self.pre_layer(x))) + x

class VallinaBackbone(nn.Module):

    def __init__(self,
                 dim, data_size, 
                 rvtm_pre_name, rvtm_pre_args,
                 vit_namex, vit_argx, vit_seqx,
                 rvtm_epi_name, rvtm_epi_args,
                 ):

        super(VallinaBackbone, self).__init__()

        # build RVTM blocks
        self.layers = nn.ModuleList([
            RVTM(dim, data_size, 
                 rvtm_pre_name, rvtm_pre_args,
                 vit_namex, vit_argx, vit_seqx[i],
                 rvtm_epi_name, rvtm_epi_args,)
            for i in range(len(vit_seqx))
        ])


        self.apply(self.initialize_weights)

    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        
        for layer in self.layers:
            x = layer(x)

        return x