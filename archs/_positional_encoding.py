import torch
from functools import lru_cache

@lru_cache()
def get_relative_position_index_3d(window_size):

    coords_d = torch.arange(window_size[0])
    coords_h = torch.arange(window_size[1])
    coords_w = torch.arange(window_size[2])
    coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # 3, Wd, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
    relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
    relative_coords[:, :, 1] += window_size[1] - 1
    relative_coords[:, :, 2] += window_size[2] - 1

    relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
    relative_coords[:, :, 1] *= (2 * window_size[2] - 1)
    relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww

    return relative_position_index