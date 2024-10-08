import math
import torch
import torch.nn as nn
import einops    
def get_sine_position_encoding_points(self, points, num_pos_feats=64, temperature=10000, normalize=True, scale=None):

        if scale is None:
            scale = 2 * math.pi


        y_embed, x_embed = points[:,:,0].unsqueeze(0), points[:,:, 1].unsqueeze(0)
        
        if normalize:
            eps = 1e-6
            y_embed = y_embed / (self.window_size[0] + eps) * scale
            x_embed = x_embed / (self.window_size[1] + eps) * scale

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=x_embed.device)
        dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / num_pos_feats)

        # BxCxHxW
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_embed = torch.cat((pos_y, pos_x), dim=3)

        return pos_embed.squeeze(0)