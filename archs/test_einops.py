import torch
from einops import repeat
import time

# Sample tensor
tensor = torch.randn(100, 100)

# Using einops repeat
start = time.time()
for _ in range(10000):
    result_einops = repeat(tensor, 'h w -> (repeat h) w', repeat=2)
einops_time = time.time() - start


# Using torch.repeat
start = time.time()
for _ in range(10000):
    result_torch = tensor.repeat(2, 1)
torch_time = time.time() - start


print(f"torch.repeat time: {torch_time:.6f} seconds")
print(f"einops.repeat time: {einops_time:.6f} seconds")
