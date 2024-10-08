import torch
import einops

# Define dimensions
B_ = 4  # batch size
C_ = 8  # number of channels
UV_ = 16  # uv size
HW_ = 10  # hw size

# Random input tensors for testing
position_bias = torch.randn(B_, UV_ * 1, C_ * 2)
feat_warp = torch.randn(B_, HW_ * UV_, C_ * 1)

# Einops-based implementation
pe_warp_einops = einops.rearrange(position_bias, "b (new uv) (c d) -> b new uv c d", c=C_, new=1)
feat_warp_einops = einops.rearrange(feat_warp, "b (hw uv) (c new) -> b hw uv c new", uv=UV_, new=1) + pe_warp_einops
feat_warp_einops = einops.rearrange(feat_warp_einops, "b hw uv c d -> b hw uv (c d)")

# Torch.view-based implementation
pe_warp_view = position_bias.view(B_, 1, UV_, C_, -1)
feat_warp_view = feat_warp.view(B_, HW_, UV_, C_, 1) + pe_warp_view
feat_warp_view = feat_warp_view.view(B_, HW_, UV_, -1)

# Check if both results are the same
if torch.allclose(feat_warp_einops, feat_warp_view):
    print("Success: Both implementations produce the same result!")
else:
    print("Error: The implementations produce different results!")

# Optional: print the differences if there is a mismatch
diff = torch.abs(feat_warp_einops - feat_warp_view)
print(f"Max difference: {diff.max()}")
