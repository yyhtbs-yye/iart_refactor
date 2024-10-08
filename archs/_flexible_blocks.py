import torch
import torch.nn as nn
import einops

class FlexibleHomoBlocks(nn.Module):

    def __init__(self, defx, argx, seqx, 
                 prev_block_type='conv2d', 
                 this_block_type='vit', 
                 post_block_type='conv2d', 
                 use_residue=False):

        super().__init__()

        # build blocks
        self.blocks = nn.ModuleList([defx[i](**argx[i]) for i in seqx])
        self.prev_block_type = prev_block_type
        self.post_block_type = post_block_type
        self.shape_table = dict(conv2d="b t c h w", vit="b t h w c", conv3d="b c t h w")
        self.prev_block_shape = self.shape_table[prev_block_type]
        self.this_block_shape = self.shape_table[this_block_type]
        self.post_block_shape = self.shape_table[post_block_type]

        self.use_residue = use_residue

    def forward(self, x):

        x = einops.rearrange(x, f"{self.prev_block_shape} -> {self.this_block_shape}")
        for blk in self.blocks:
            # Apply the block with optional residual connection
            if self.use_residue:
                x = blk(x) + x
            else:
                x = blk(x)
        x = einops.rearrange(x, f"{self.this_block_shape} -> {self.post_block_shape}")

        return x

class FlexibleHeteBlocks(nn.Module):

    def __init__(self, defx, argx, seqx, 
                 prev_block_type='conv2d', 
                 this_block_types=None, 
                 post_block_type='conv2d', 
                 use_residue=False):

        super().__init__()

        # Validate that all lists are of the same length
        assert len(defx) == len(argx) == len(this_block_types), "defx, argx, and block_types must have the same length"
        
        # Store block functions, arguments, and block types
        self.blocks = nn.ModuleList([defx[i](**argx[i]) for i in seqx])
        self.this_block_types = [this_block_types[i] for i in seqx]

        # Define a shape table for block types
        self.shape_table = dict(conv2d="b t c h w", vit="b t h w c", conv3d="b c t h w")
        self.prev_block_type = prev_block_type
        self.post_block_type = post_block_type

        self.use_residue = use_residue

    def forward(self, x):
        
        # Initialize the current block type to the first one
        current_block_type = self.prev_block_type

        # Iterate through the blocks and apply each one
        for i, blk in enumerate(self.blocks):
            next_block_type = self.this_block_types[i]

            # Check if block type changes and rearrange the tensor shape if necessary
            if next_block_type != current_block_type:
                next_block_shape = self.shape_table[next_block_type]
                current_block_shape = self.shape_table[current_block_type]
                x = einops.rearrange(x, f"{current_block_shape} -> {next_block_shape}")
                current_block_type = next_block_type

            # Apply the block with optional residual connection
            if self.use_residue:
                x = blk(x) + x
            else:
                x = blk(x)

        current_block_shape = self.shape_table[current_block_type]
        post_block_shape = self.shape_table[self.post_block_type]
        x = einops.rearrange(x, f"{current_block_shape} -> {post_block_shape}")

        return x
