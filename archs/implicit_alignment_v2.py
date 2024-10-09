import math
import torch
import torch.nn as nn
import einops

def gather_hw(x, h_idx, w_idx, B_, C_, H_, W_):

    x = x.contiguous()

    linear_idx = w_idx + W_ * h_idx + H_ * W_ * torch.arange(B_, dtype=h_idx.dtype, device=h_idx.device).view(-1, 1)
    linear_idx = linear_idx.view(-1)

    x = einops.rearrange(x, "b c h w -> c (b h w)")

    y = x[:, linear_idx]

    return einops.rearrange(y, "c (b hwuv) -> b hwuv c", b=B_, c=C_)

def get_sine_position_encoding(y_embed, x_embed, num_pos_feats=64, 
                               temperature=10000, use_norm=True, 
                               scale=2 * math.pi, card=None):

        if use_norm:
            eps = 1e-6
            y_embed = y_embed / (card[0] + eps) * scale
            x_embed = x_embed / (card[1] + eps) * scale

        # y_embed.shape=(b h w)
        dim_t = torch.arange(num_pos_feats, dtype=x_embed.dtype, device=x_embed.device)
        dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / num_pos_feats)

        # b h w -> b h w p
        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        
        # b h w p -> b (h/2+h/2) (w/2+w/2) p
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        
        # (b h w p, b h w p) -> b h w p*2
        pos_embed = torch.cat((pos_y, pos_x), dim=-1)

        # pos_embed = einops.rearrange(pos_embed, 'b h w p -> b (h w) p')

        return pos_embed

class ImplicitWarpModule(nn.Module):

    def __init__(self,
                 dim, image_size, window_size=2,
                 pe_wrp=True, pe_x=True,
                 pe_dim = 128, pe_temp = 10000,
                 warp_padding='duplicate',
                 num_heads=8,
                 aux_loss_out = False, aux_loss_dim = 3,
                 qkv_bias=True, qk_scale=None,
                 use_corrected_card=False,
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
        
        # self.n_window_pixels = self.window_size[0] * self.window_size[1]

        y_embed, x_embed = torch.meshgrid(torch.arange(self.window_size[0], dtype=torch.float32), 
                                          torch.arange(self.window_size[1], dtype=torch.float32), indexing='ij')
        
        y_embed = y_embed.flatten().unsqueeze(0)
        x_embed = x_embed.flatten().unsqueeze(0)

        card = self.window_size if use_corrected_card else [it - 1 for it in self.window_size]

        self.register_buffer("position_bias", get_sine_position_encoding(y_embed, x_embed, 
                                                                         pe_dim // 2, temperature=self.pe_temp, 
                                                                         use_norm=True, card=card)) 

        self.register_buffer("window_idx_offset", torch.stack(torch.meshgrid(
                                                              torch.arange(0, self.window_size[0], dtype=int),
                                                              torch.arange(0, self.window_size[1], dtype=int)), 2))

        # self.register_buffer("image_idx_offset", torch.stack(torch.meshgrid(
        #                                                      torch.arange(0, self.image_size[0], dtype=int),
        #                                                      torch.arange(0, self.image_size[1], dtype=int)), 2))


        self.initialize_weights()

    def initialize_weights(self):
        # Xavier/Glorot initialization for q, k, v weights
        nn.init.xavier_uniform_(self.q.weight)
        nn.init.xavier_uniform_(self.k.weight)
        nn.init.xavier_uniform_(self.v.weight)

        # Optionally, initialize the bias terms with zero or small values
        if self.q.bias is not None:
            nn.init.zeros_(self.q.bias)
        if self.k.bias is not None:
            nn.init.zeros_(self.k.bias)
        if self.v.bias is not None:
            nn.init.zeros_(self.v.bias)

        # Initialize projection layer (for auxiliary output), if exists
        if self.aux_loss_out:
            nn.init.xavier_uniform_(self.proj.weight)
            if self.proj.bias is not None:
                nn.init.zeros_(self.proj.bias)

    def forward(self, feat_supp, feat_curr, flow):
        # feat_supp: frame to be propagated.
        # feat_curr: frame propagated to.
        # flow: optical flow from feat_curr to feat_supp 
        B_, C_, H_, W_ = feat_curr.size()
        U_, V_ = self.window_size
        UV_ = U_ * V_
        HW_ = H_ * W_
        # Computes the total number of pixels (flattened) across the batch.
        BHW_ = B_ * H_ * W_

        # Create the Image Index
        image_idx_offset = torch.stack(torch.meshgrid(torch.arange(0, H_, dtype=int),
                                                      torch.arange(0, W_, dtype=int)), 2).to(feat_supp.device)

        # The flow field (flow) is flipped along its last dimension, then added to 
        # the mesh `grid_orig`. This creates the "warped grid_orig" (grid_wrp_full), 
        # where each `grid_orig` location in feat_curr is displaced according to the flow.
        grid_wrp_full = image_idx_offset.unsqueeze(0) + flow.flip(dims=(-1,))

        # The warped grid_orig grid_wrp_full is decomposed into two parts:
        # grid_wrp_intg: the integer part of the warped coordinates (by flooring).
        # grid_wrp_deci: the fractional part, representing sub-pixel displacements.
        grid_wrp_intg = torch.floor(grid_wrp_full).int()
        grid_wrp_deci = grid_wrp_full - grid_wrp_intg

        # grid_wrp_intg.shape=(B_ H_, W_, 2) -> grid_wrp_intg.unsqueeze(2).shape=(B_ H_, W_, 1, 2)
        grid_wrp_full = grid_wrp_intg.unsqueeze(3).unsqueeze(3) + self.window_idx_offset.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        grid_wrp_full = grid_wrp_full.reshape(B_, -1, 2) # (B_ H_, W_, wh, ww, 2) -> (B_ H_*W_*wh*ww, 2)
        grid_wrp_deci = grid_wrp_deci.reshape(B_, -1, 2) # (B_ H_, W_, 2) -> (B_ H_*W_, 2)

#*-----*# ---Integer Resampling--------------------------------------------------------------------
        # If the padding method is "duplicate", out-of-bound coordinates are clamped to 
        # the valid range (i.e., pixel indices are constrained to [0, H_-1] for height 
        # and [0, W_-1] for width).
        if self.warp_padding == 'duplicate':
            h_idx = grid_wrp_full[..., 0].clamp(min=0, max=H_-1)
            w_idx = grid_wrp_full[..., 1].clamp(min=0, max=W_-1)
            feat_warp = gather_hw(feat_supp, h_idx, w_idx, B_, C_, H_, W_)

        # In this case, if "zero" padding is used, the code checks for out-of-bound pixel 
        # indices (invalid locations) and sets the corresponding gathered values to zero.
        elif self.warp_padding == 'zero':
            invalid_h = torch.logical_or(grid_wrp_full[:,:,0]<0, grid_wrp_full[:,:,0]>H_-1)
            invalid_w = torch.logical_or(grid_wrp_full[:,:,1]<0, grid_wrp_full[:,:,1]>H_-1)
            invalid = torch.logical_or(invalid_h, invalid_w)

            h_idx = grid_wrp_full[:,:,0].clamp(min=0, max=H_-1)
            w_idx = grid_wrp_full[:,:,1].clamp(min=0, max=W_-1)
            feat_warp = gather_hw(feat_supp, h_idx, w_idx, B_, C_, H_, W_)

            feat_warp[invalid] = 0
        else:
            raise ValueError(f'self.warp_padding: {self.warp_padding}')
#*-----*# ---Decimal Resampling--------------------------------------------------------------------

        pe_warp = einops.rearrange(self.position_bias, "b (new uv) (c d) -> b new uv c d", c=C_, new=1)
        feat_warp = einops.rearrange(feat_warp, "b (hw uv) (c new) -> b hw uv c new", uv=U_*V_, new=1) + pe_warp
        feat_warp = einops.rearrange(feat_warp, "b hw uv c d -> b hw uv (c d)")

        # pe_warp = self.position_bias.view(1, 1, UV_, C_, -1)
        # feat_warp = feat_warp.view(B_, HW_, UV_, C_, 1) + pe_warp
        # feat_warp = feat_warp.reshape(B_, HW_*UV_, -1)

        # (b c h w) -> (b c hw) -> (b hw c)
        feat_curr = feat_curr.flatten(2).permute(0, 2, 1)
        
        pe_curr = get_sine_position_encoding(grid_wrp_deci[..., 0], 
                                             grid_wrp_deci[..., 1], 
                                             self.pe_dim // 2, temperature=self.pe_temp,
                                             use_norm=True, card=self.window_size)

        pe_curr = einops.rearrange(pe_curr, "b hw (c d) -> b hw c d", c=C_)
        feat_curr = einops.rearrange(feat_curr, "b hw (c new) -> b hw c new", new=1) + pe_curr
        feat_curr = einops.rearrange(feat_curr, "b hw c d -> b hw (c d)")

        # pe_curr = pe_curr.view(B_, HW_, C_, -1)
        # feat_curr = feat_curr.view(B_, HW_,  C_, 1) + pe_curr
        # feat_curr = feat_curr.reshape(B_, HW_, -1)

        kw = self.k(feat_warp).reshape(BHW_, UV_, self.num_heads, self.dim // self.num_heads).permute(0, 2, 1, 3) 
        vw = self.v(feat_warp).reshape(BHW_, UV_, self.num_heads, self.dim // self.num_heads).permute(0, 2, 1, 3)
        qx = self.q(feat_curr).reshape(BHW_, self.num_heads, self.dim // self.num_heads).unsqueeze(1).permute(0, 2, 1, 3)

        # kw/vw.shape=(BHW_, n_heads, win, d)
        # qx.shape=(BHW_, n_heads, 1, d)
        attn = (qx * self.scale) @ kw.transpose(-2, -1)

        # attn.shape=(BHW_, n_heads, 1, w)
        attn = self.softmax(attn)

        # (attn @ vw).shape=(BHW_, n_heads, 1, d)
        # (attn @ vw).transpose(1, 2).shape=(BHW_, 1, n_heads, d)
        # reshape -> out.shape=(BHW_, 1, n_heads*d)
        out = (attn @ vw).transpose(1, 2).reshape(BHW_, 1, self.dim)
        # squeeze() -> out.shape=(BHW_, n_heads*d)=(BHW_, n_heads*d)
        out = out.squeeze(1)
        
        # If auxiliary loss output is required, the output is projected back to
        if self.aux_loss_out:
            out_rgb = self.proj(out).reshape(B_, H_, W_, C_).permute(0, 3, 1, 2)
            return out.reshape(B_, H_, W_, self.dim).permute(0, 3, 1, 2), out_rgb
        else:
            return out.reshape(B_, H_, W_, self.dim).permute(0, 3, 1, 2)

# Call the test function
if __name__ == "__main__":
    # Define dimensions for the test
    dim = 128  # Dimension of the features
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
        image_size=image_size,
        window_size=window_size,
        pe_dim=pe_dim,
        num_heads=num_heads,
        aux_loss_out=aux_loss_out,
        aux_loss_dim=aux_loss_dim
    ).to('cuda:2')

    seed = 42
    torch.manual_seed(seed)

    # Generate random input data
    feat_supp = torch.rand(batch_size, dim, image_size[0], image_size[1]).to('cuda:2')
    feat_curr = torch.rand(batch_size, dim, image_size[0], image_size[1]).to('cuda:2')
    flow = torch.rand(batch_size, image_size[0], image_size[1], 2).to('cuda:2')  # Flow with 2 channels (x and y displacement)

    import time
    start_time = time.time()

    # Forward pass through the model
    for i in range(50):
        output = model(feat_supp, feat_curr, flow)
    
    print("--- %s seconds ---" % (time.time() - start_time))
    # print("output", output)

    # Print the output shapes to verify the functionality
    if aux_loss_out:
        assert len(output) == 2
        out_features, aux_out = output
        print("Output shape:", out_features.shape)              # Expected: [batch_size, dim, H_, W_]
        print("Auxiliary loss output shape:", aux_out.shape)    # Expected: [batch_size, aux_loss_dim, H_, W_]
    else:
        print("Output shape:", output.shape)                    # Expected: [batch_size, dim, H_, W_]
