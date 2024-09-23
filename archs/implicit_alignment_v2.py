import math
import torch
import torch.nn as nn
import einops

class ImplicitWarpModule(nn.Module):

    def __init__(self,
                 dim, 
                 image_size,
                 window_size=2,
                 pe_wrp=True,
                 pe_x=True,
                 pe_dim = 48,
                 pe_temp = 10000,
                 warp_padding='duplicate',
                 num_heads=8,
                 aux_loss_out = False,
                 aux_loss_dim = 3,
                 qkv_bias=True,
                 qk_scale=None,
                 use_checkpoint_attn=False,
                 use_checkpoint_ffn=False,
                 ):
        
        super().__init__()
        self.dim = dim
        self.use_pe = pe_wrp
        self.pe_x = pe_x
        self.pe_dim = pe_dim
        self.pe_temp = pe_temp
        self.aux_loss_out = aux_loss_out

        self.num_heads = num_heads
        assert self.dim % self.num_heads == 0
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.window_size = (window_size, window_size)
        self.image_size = image_size
        self.warp_padding = warp_padding
        self.q = nn.Linear(pe_dim, dim, bias=qkv_bias)
        self.k = nn.Linear(pe_dim, dim, bias=qkv_bias)
        self.v = nn.Linear(pe_dim, dim, bias=qkv_bias)
        if self.aux_loss_out:
            self.proj = nn.Linear(dim, aux_loss_dim)

        self.softmax = nn.Softmax(dim=-1)
        
        self.n_window_pixels = self.window_size[0] * self.window_size[1]

        self.register_buffer("position_bias", self.get_sine_position_encoding(self.window_size, pe_dim // 2, temperature=self.pe_temp, normalize=True))


        self.register_buffer("window_idx_offset", torch.stack(torch.meshgrid(
                                                                torch.arange(0, self.window_size[0], dtype=int),
                                                                torch.arange(0, self.window_size[1], dtype=int)
                                                                ), 2).reshape(self.n_window_pixels, 2))

        self.register_buffer("image_idx_offset", torch.stack(torch.meshgrid(
                                                                torch.arange(0, self.image_size[0], dtype=int),
                                                                torch.arange(0, self.image_size[1], dtype=int)
                                                                ), 2))


    def gather_hw(self, x, h_idx, w_idx):
        # Linearize the last two dims and index in a contiguous x
        x = x.contiguous()
        lin_idx = w_idx + x.size(-1) * h_idx                                  # Linear Index Calculation
        x = x.view(-1, x.size(1), x.size(2) * x.size(3))                    # Reshape Tensor to B, T, H*W
        return x.gather(-1, lin_idx.unsqueeze(1).repeat(1,x.size(1),1))     # Gather on H*W dimension according to given [Hid, Wid]


    def forward(self, feat_supp, feat_curr, flow):
        # feat_supp: frame to be propagated.
        # feat_curr: frame propagated to.
        # flow: optical flow from feat_curr to feat_supp 
        n, c, h, w = feat_curr.size()

        # The flow field (flow) is flipped along its last dimension, then added to 
        # the mesh grid_orig. This creates the "warped grid_orig" (grid_wrp_full), where each grid_orig 
        # location in feat_curr is displaced according to the flow.
        grid_wrp_full = self.image_idx_offset.unsqueeze(0) + flow.flip(dims=(-1,))

        # The warped grid_orig grid_wrp_full is decomposed into two parts:
        # grid_wrp_int: the integer part of the warped coordinates (by flooring).
        # grid_wrp_dec: the fractional part, representing sub-pixel displacements.
        grid_wrp_int = torch.floor(grid_wrp_full).int()
        grid_wrp_dec = grid_wrp_full - grid_wrp_int


        # Both grid_wrp_int and grid_wrp_dec are reshaped from (n, h, w, 2) to 
        # (n, h*w, 2) for easier processing in the next steps.
        grid_wrp_int = grid_wrp_int.reshape(n, h*w, 2)
        grid_wrp_dec = grid_wrp_dec.reshape(n, h*w, 2)

        ## Get small 4x4 windows around the integer grid_orig coordinates in the reference frame
        # self.window_idx_offset.shape=(self.wh*ww, 2)
        #     -> self.window_idx_offset.unsqueeze(0).unsqueeze(0).shape=(1, 1, wh*ww, 2)
        # grid_wrp_int.shape=(n, h*w, 2) -> grid_wrp_int.unsqueeze(2).shape=(n, h*w, 1, 2)
        grid_wrp_full = grid_wrp_int.unsqueeze(2) + self.window_idx_offset.unsqueeze(0).unsqueeze(0)
        # grid_wrp_full.shape=(n, h*w, wh*ww, 2) -> (n, h*w*wh*ww, 2)
        grid_wrp_full = grid_wrp_full.reshape(n, -1, 2)
        # WARNING!!! grid_wrp_full is HUGE in memory!!!



#*-----*# ---Integer Resampling--------------------------------------------------------------------
        # If the padding method is "duplicate", out-of-bound coordinates are clamped to 
        # the valid range (i.e., pixel indices are constrained to [0, h-1] for height 
        # and [0, w-1] for width).
        if self.warp_padding == 'duplicate':
            h_idx = grid_wrp_full[:,:,0].clamp(min=0, max=h-1)
            w_idx = grid_wrp_full[:,:,1].clamp(min=0, max=w-1)
            #---# The gathered values are then reshaped into the required format for further computation
            feat_warp = einops.rearrange(
                self.gather_hw(feat_supp, h_idx, w_idx),
                'n c (h w) nwp -> n (h w nwp) c'
            )

        # In this case, if "zero" padding is used, the code checks for out-of-bound pixel 
        # indices (invalid locations) and sets the corresponding gathered values to zero.
        elif self.warp_padding == 'zero':
            invalid_h = torch.logical_or(grid_wrp_full[:,:,0]<0, grid_wrp_full[:,:,0]>h-1)
            invalid_w = torch.logical_or(grid_wrp_full[:,:,1]<0, grid_wrp_full[:,:,1]>h-1)
            invalid = torch.logical_or(invalid_h, invalid_w)

            h_idx = grid_wrp_full[:,:,0].clamp(min=0, max=h-1)
            w_idx = grid_wrp_full[:,:,1].clamp(min=0, max=w-1)

            feat_warp = einops.rearrange(
                self.gather_hw(feat_supp, h_idx, w_idx),
                'n c (h w) nwp -> n (h w nwp) c'
            )

            feat_warp[invalid] = 0
        else:
            raise ValueError(f'self.warp_padding: {self.warp_padding}')
        
#*-----*# ---Decimal Resampling--------------------------------------------------------------------
        # Positional encoding (a bias term representing spatial positions, such as 
        # sine and cosine encoding) is repeated across all pixels to align with the 
        # warped grid_orig.
        # self.position_bias.shape=(1, wH * wW, 2 * num_pos_feats)
        # pe_warp.shape=(n, wH*wW*h*w, 2 * num_pos_feats)
        pe_warp = self.position_bias.repeat(n, h * w, 1)

        # Adds positional encoding to the windowed patches, depending on whether 
        # `self.use_pe` is set. The positional encoding may be used to improve 
        # spatial awareness in the cross-attention mechanism.
        if self.use_pe:
            feat_warp = feat_warp.repeat(1, 1, self.pe_dim // c) + pe_warp
        else:
            feat_warp = feat_warp.repeat(1, 1, self.pe_dim // c)

        # Flattens the tensor `feat_curr` and applies positional encoding (sine and cosine 
        # encoding) to the source pixel based on the fractional offsets 
        # `grid_wrp_dec`.

        feat_curr = feat_curr.flatten(2).permute(0,2,1)
        pe_curr = self.get_sine_position_encoding_points(grid_wrp_dec, self.pe_dim // 2, temperature=self.pe_temp, normalize=True)

        # Adds the positional encoding to `feat_curr`, depending on whether `self.pe_x` is set.
        if self.pe_x:
            feat_curr = feat_curr.repeat(1, 1, self.pe_dim // c) + pe_curr
        else:
            feat_curr = feat_curr.repeat(1, 1, self.pe_dim // c)

        # Computes the total number of pixels (flattened) across the batch.
        nhw = n * h * w
        
        # Applies learnable linear transformations `self.k`, `self.v`, and `self.q` 
        # to generate the key, value, and query tensors used in the attention 
        # mechanism. The tensors are reshaped for multi-head attention.

        kw = self.k(feat_warp).reshape(nhw, self.n_window_pixels, self.num_heads, self.dim // self.num_heads).permute(0, 2, 1, 3) 
        vw = self.v(feat_warp).reshape(nhw, self.n_window_pixels, self.num_heads, self.dim // self.num_heads).permute(0, 2, 1, 3)
        qx = self.q(feat_curr).reshape(nhw, self.num_heads, self.dim // self.num_heads).unsqueeze(1).permute(0, 2, 1, 3)

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

        # not_mask: (1, H, W)
        not_mask = torch.ones([1, HW[0], HW[1]])

        
        y_embed = not_mask.cumsum(1, dtype=torch.float32) - 1
        x_embed = not_mask.cumsum(2, dtype=torch.float32) - 1
        if normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale
        
        # x_embed,y_embed.shape=(1, H, W)

        # dim_t.shape=(num_pos_feats,)
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / num_pos_feats)

        # pos_x.shape=(1, H, W, num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        # (1, 2 * num_pos_feats, H, W)
        pos_embed = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        # (1, H * W, 2 * num_pos_feats)
        return pos_embed.flatten(2).permute(0, 2, 1).contiguous()

        