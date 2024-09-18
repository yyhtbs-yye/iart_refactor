import torch
import numpy as np
from functools import lru_cache

def window_partition(x, window_size):
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size

    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2],
               window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0] * window_size[1] * window_size[2],
                                                                  C)
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1],
                     window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x

def compute_padding(window_size, t, h, w):
    pad_t0 = pad_h0 = pad_w0 = 0
    pad_t1 = (window_size[0] - t % window_size[0]) % window_size[0]
    pad_h1 = (window_size[1] - h % window_size[1]) % window_size[1]
    pad_w1 = (window_size[2] - w % window_size[2]) % window_size[2]
    return pad_w0, pad_w1, pad_h0, pad_h1, pad_t0, pad_t1, 

def patch_partition(x, patch_size):
    """
    Partition video into non-overlapping patches.
    
    Args:
        x: Input tensor of shape (b, d, h, w, c) - (batch, depth (time), height, width, channels)
        patch_size: Tuple (pt, ph, pw) - patch size for (depth, height, width)
    
    Returns:
        patches: Tensor of shape (b, num_patches, patch_size_depth * patch_size_height * patch_size_width * c)
    """
    B, D, H, W, C = x.shape
    patch_depth, patch_height, patch_width = patch_size
    
    # Reshape into patches of size (patch_depth, patch_height, patch_width) for each video
    x = x.view(B, D // patch_depth, patch_depth, H // patch_height, patch_height, W // patch_width, patch_width, C)
    
    # Rearrange dimensions so that patches are contiguous and flatten each patch into a single vector
    patches = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()  # Shape: (b, d//pt, h//ph, w//pw, pt, ph, pw, c)
    patches = patches.view(B, -1, patch_depth * patch_height * patch_width * C)  # Shape: (b, num_patches, patch_size * c)
    
    return patches

def patch_reverse(patches, patch_size, B, D, H, W, C):
    """
    Reverse the patch partition operation by reconstructing the video from patches.
    
    Args:
        patches: Tensor of shape (b, num_patches, patch_size_depth * patch_size_height * patch_size_width * c)
        patch_size: Tuple (pt, ph, pw) - patch size for (depth, height, width)
        video_size: Tuple (D, H, W, C) - original video size (depth, height, width, channels)
    
    Returns:
        x: Reconstructed video of shape (b, D, H, W, C)
    """
    patch_depth, patch_height, patch_width = patch_size

    # Calculate the number of patches along each dimension
    num_depth_patches = D // patch_depth
    num_height_patches = H // patch_height
    num_width_patches = W // patch_width

    # Reshape patches back into (B, num_depth_patches, num_height_patches, num_width_patches, pt, ph, pw, c)
    patches = patches.view(B, num_depth_patches, num_height_patches, num_width_patches,
                           patch_depth, patch_height, patch_width, C)

    # Permute to bring patches back into the original video dimensions
    x = patches.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    
    # Reshape into the full video size
    x = x.view(B, D, H, W, C)  # Shape: (B, D, H, W, C)

    return x
