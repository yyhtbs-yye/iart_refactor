import math
import torch
import torch.nn as nn

class ImplicitWarpModule(nn.Module):
    """ Implicit Warp Module.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for mutual and self attention.
        mut_attn (bool): If True, use mutual and self attention. Default: True.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True.
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm.
        use_checkpoint_attn (bool): If True, use torch.checkpoint for attention modules. Default: False.
        use_checkpoint_ffn (bool): If True, use torch.checkpoint for feed-forward modules. Default: False.
    """

    def __init__(self,
                 dim,
                 pe_wrp=True,
                 pe_x=True,
                 pe_dim = 48,
                 pe_temp = 10000,
                 warp_padding='duplicate',
                 num_heads=8,
                 aux_loss_out = False,
                 aux_loss_dim = 3,
                 window_size=2,
                 qkv_bias=True,
                 qk_scale=None,
                 use_checkpoint_attn=False,
                 use_checkpoint_ffn=False,
                 ):
        super().__init__()
        self.dim = dim
        self.pe_wrp = pe_wrp
        self.pe_x = pe_x
        self.pe_dim = pe_dim
        self.pe_temp = pe_temp
        self.aux_loss_out = aux_loss_out

        self.num_heads = num_heads
        assert self.dim % self.num_heads == 0
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.window_size = (window_size, window_size)
        self.warp_padding = warp_padding
        self.q = nn.Linear(pe_dim, dim, bias=qkv_bias)
        self.k = nn.Linear(pe_dim, dim, bias=qkv_bias)
        self.v = nn.Linear(pe_dim, dim, bias=qkv_bias)
        if self.aux_loss_out:
            self.proj = nn.Linear(dim, aux_loss_dim)

        self.softmax = nn.Softmax(dim=-1)
        
        self.register_buffer("position_bias", self.get_sine_position_encoding(self.window_size, pe_dim // 2, temperature=self.pe_temp, normalize=True))

        grid_h, grid_w = torch.meshgrid(
            torch.arange(0, self.window_size[0], dtype=int),
            torch.arange(0, self.window_size[1], dtype=int)
        )

        self.num_values = self.window_size[0] * self.window_size[1]

        self.register_buffer("window_idx_offset", torch.stack((grid_h, grid_w), 2).reshape(self.num_values, 2))

        self.initialize_weights()

    def initialize_weights(self):
        # Set weights and biases of self.q, self.k, self.v to 1
        nn.init.constant_(self.q.weight, 1)
        if self.q.bias is not None:
            nn.init.constant_(self.q.bias, 1)

        nn.init.constant_(self.k.weight, 1)
        if self.k.bias is not None:
            nn.init.constant_(self.k.bias, 1)

        nn.init.constant_(self.v.weight, 1)
        if self.v.bias is not None:
            nn.init.constant_(self.v.bias, 1)

        if self.aux_loss_out:
            nn.init.constant_(self.proj.weight, 1)
            if self.proj.bias is not None:
                nn.init.constant_(self.proj.bias, 1)

    def gather_hw(self, x, idx1, idx2):
        # Linearize the last two dims and index in a contiguous x
        x = x.contiguous()
        lin_idx = idx2 + x.size(-1) * idx1                                  # Linear Index Calculation
        x = x.view(-1, x.size(1), x.size(2) * x.size(3))                    # Reshape Tensor to B, T, H*W
        return x.gather(-1, lin_idx.unsqueeze(1).repeat(1,x.size(1),1))     # Gather on H*W dimension according to given [Hid, Wid]


    def forward(self, y, x, flow):
        # y: frame to be propagated.
        # x: frame propagated to.
        # flow: optical flow from x to y 
        if x.size()[-2:] != flow.size()[1:3]:
            raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                            f'flow ({flow.size()[1:3]}) are not the same.')
        n, c, h, w = x.size()

        # create mesh grid
        device = flow.device

        # Creates a 2D mesh grid for the height (grid_h) and width (grid_w) dimensions. 
        # This mesh grid is used to map the positions in the source image to the warped 
        # positions in the reference image based on the optical flow.
        grid_h, grid_w = torch.meshgrid(
            torch.arange(0, h, device=device, dtype=x.dtype),
            torch.arange(0, w, device=device, dtype=x.dtype)
        )


        # Combines (stack) grid_h and grid_w to form a grid of shape (h, w, 2) (where
        # the last dimension stores both coordinates). The grid is then repeated for
        # all elements in the batch. It is set to not require gradient computation
        # (requires_grad=False).
        grid = torch.stack((grid_h, grid_w), 2).repeat(n, 1, 1, 1)  # h, w, 2
        grid.requires_grad = False


        # The flow field (flow) is flipped along its last dimension, then added to 
        # the mesh grid. This creates the "warped grid" (grid_wrp), where each grid 
        # location in x is displaced according to the flow.
        grid_wrp = grid + flow.flip(dims=(-1,)) # grid_wrp


        # The warped grid grid_wrp is decomposed into two parts:
        # grid_wrp_flr: the integer part of the warped coordinates (by flooring).
        # grid_wrp_off: the fractional part, representing sub-pixel displacements.
        grid_wrp_flr = torch.floor(grid_wrp).int()
        grid_wrp_off = grid_wrp - grid_wrp_flr


        # Both grid_wrp_flr and grid_wrp_off are reshaped from (n, h, w, 2) to 
        # (n, h*w, 2) for easier processing in the next steps.
        grid_wrp_flr = grid_wrp_flr.reshape(n, h*w, 2)
        grid_wrp_off = grid_wrp_off.reshape(n, h*w, 2)

        ## Get small 4x4 windows around the integer grid coordinates in the reference frame
        grid_wrp = grid_wrp_flr.unsqueeze(2).repeat(1, 1, self.num_values, 1) + self.window_idx_offset 
        grid_wrp = grid_wrp.reshape(n, h*w*self.num_values, 2)

        # If the padding method is "duplicate", out-of-bound coordinates are clamped to 
        # the valid range (i.e., pixel indices are constrained to [0, h-1] for height 
        # and [0, w-1] for width).
        if self.warp_padding == 'duplicate':
            idx0 = grid_wrp[:,:,0].clamp(min=0, max=h-1)
            idx1 = grid_wrp[:,:,1].clamp(min=0, max=w-1)
            #---# The gathered values are then reshaped into the required format for further computation
            wrp = self.gather_hw(y, idx0, idx1).reshape(n, c, h*w, self.num_values).permute(0,2,3,1).reshape(n, h*w*self.num_values, c)
        # In this case, if "zero" padding is used, the code checks for out-of-bound pixel 
        # indices (invalid locations) and sets the corresponding gathered values to zero.
        elif self.warp_padding == 'zero':
            invalid_h = torch.logical_or(grid_wrp[:,:,0]<0, grid_wrp[:,:,0]>h-1)
            invalid_w = torch.logical_or(grid_wrp[:,:,1]<0, grid_wrp[:,:,1]>h-1)
            invalid = torch.logical_or(invalid_h, invalid_w)

            idx0 = grid_wrp[:,:,0].clamp(min=0, max=h-1)
            idx1 = grid_wrp[:,:,1].clamp(min=0, max=w-1)

            wrp = self.gather_hw(y, idx0, idx1).reshape(n, c, h*w, self.num_values).permute(0,2,3,1).reshape(n, h*w*self.num_values, c)
            wrp[invalid] = 0
        else:
            raise ValueError(f'self.warp_padding: {self.warp_padding}')
        
        # Positional encoding (a bias term representing spatial positions, such as 
        # sine and cosine encoding) is repeated across all pixels to align with the 
        # warped grid.
        wrp_pe = self.position_bias.repeat(n, h*w, 1)

        # Adds positional encoding to the windowed patches, depending on whether 
        # `self.pe_wrp` is set. The positional encoding may be used to improve 
        # spatial awareness in the cross-attention mechanism.
        if self.pe_wrp:
            wrp = wrp.repeat(1,1,self.pe_dim//c) + wrp_pe
        else:
            wrp = wrp.repeat(1,1,self.pe_dim//c)

        # Flattens the tensor `x` and applies positional encoding (sine and cosine 
        # encoding) to the source pixel based on the fractional offsets 
        # `grid_wrp_off`.

        x = x.flatten(2).permute(0,2,1)
        x_pe = self.get_sine_position_encoding_points(grid_wrp_off, self.pe_dim // 2, temperature=self.pe_temp, normalize=True)

        # Adds the positional encoding to `x`, depending on whether `self.pe_x` is set.
        if self.pe_x:
            x = x.repeat(1,1,self.pe_dim//c) + x_pe
        else:
            x = x.repeat(1,1,self.pe_dim//c)

        # Computes the total number of pixels (flattened) across the batch.
        nhw = n*h*w
        
        # Applies learnable linear transformations `self.k`, `self.v`, and `self.q` 
        # to generate the key, value, and query tensors used in the attention 
        # mechanism. The tensors are reshaped for multi-head attention.

        kw = self.k(wrp).reshape(nhw, self.num_values, self.num_heads, self.dim // self.num_heads).permute(0, 2, 1, 3) 
        vw = self.v(wrp).reshape(nhw, self.num_values, self.num_heads, self.dim // self.num_heads).permute(0, 2, 1, 3)
        qx = self.q(x).reshape(nhw, self.num_heads, self.dim // self.num_heads).unsqueeze(1).permute(0, 2, 1, 3)

        # The query (qx) is scaled and matrix-multiplied with the transposed key 
        # (kw) to compute attention weights, which are then normalized using 
        # softmax.
        attn = (qx * self.scale) @ kw.transpose(-2, -1)
        attn = self.softmax(attn)

        # The attention weights are applied to the value tensor vw, and the output 
        # is reshaped back to the original form.
        out = (attn @ vw).transpose(1, 2).reshape(nhw, 1, self.dim)
        out = out.squeeze(1)
        
        # If auxiliary loss output is required, the output is projected back to
        if self.aux_loss_out:
            out_rgb = self.proj(out).reshape(n, h, w, c).permute(0,3,1,2)
            return out.reshape(n, h, w, self.dim).permute(0,3,1,2), out_rgb
        else:
            return out.reshape(n, h, w, self.dim).permute(0,3,1,2)

    def get_sine_position_encoding_points(self, points, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
        """ get_sine_position_encoding_points for single points.

        Args:
            points (tuple[int]): The temporal length, height and width of the window.
            num_pos_feats
            temperature
            normalize
            scale
            num_heads (int): Number of attention heads.
            qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
            qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
            mut_attn (bool): If True, add mutual attention to the module. Default: True
        """

        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")

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

    def get_sine_position_encoding(self, HW, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
        """ Get sine position encoding """
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")

        if scale is None:
            scale = 2 * math.pi

        not_mask = torch.ones([1, HW[0], HW[1]])
        y_embed = not_mask.cumsum(1, dtype=torch.float32) - 1
        x_embed = not_mask.cumsum(2, dtype=torch.float32) - 1
        if normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / num_pos_feats)

        # BxCxHxW
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_embed = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos_embed.flatten(2).permute(0, 2, 1).contiguous()


# Call the test function
if __name__ == "__main__":
    # Define dimensions for the test
    dim = 384  # Dimension of the features
    image_size = (256, 256)  # Image size (height, width)
    window_size = 4  # Window size for positional encoding
    pe_dim = 256  # Positional encoding dimension
    num_heads = 8  # Number of attention heads
    aux_loss_out = False  # Whether to output auxiliary loss
    aux_loss_dim = 3  # Dimension of auxiliary loss output
    batch_size = 2  # Batch size for the test

    # Create the ImplicitWarpModule instance
    model = ImplicitWarpModule(
        dim=dim,
        window_size=window_size,
        pe_dim=pe_dim,
        num_heads=num_heads,
        aux_loss_out=aux_loss_out,
        aux_loss_dim=aux_loss_dim
    ).to('cuda:3')

    seed = 42
    torch.manual_seed(seed)

    # Generate random input data
    feat_supp = torch.rand(batch_size, dim, image_size[0], image_size[1]).to('cuda:3')
    feat_curr = torch.rand(batch_size, dim, image_size[0], image_size[1]).to('cuda:3')
    flow = torch.rand(batch_size, image_size[0], image_size[1], 2).to('cuda:3')  # Flow with 2 channels (x and y displacement)

    import time
    start_time = time.time()

    # Forward pass through the model
    for i in range(50):
        output = model(feat_supp, feat_curr, flow)
    
    print("--- %s seconds ---" % (time.time() - start_time))

    print("output", output)

    # Print the output shapes to verify the functionality
    if aux_loss_out:
        assert len(output) == 2
        out_features, aux_out = output
        print("Output shape:", out_features.shape)  # Expected: [batch_size, dim, H_, W_]
        print("Auxiliary loss output shape:", aux_out.shape)  # Expected: [batch_size, aux_loss_dim, H_, W_]
    else:
        print("Output shape:", output.shape)  # Expected: [batch_size, dim, H_, W_]
