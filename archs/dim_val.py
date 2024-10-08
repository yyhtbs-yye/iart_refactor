import torch

# Define dimensions
B_ = 4  # Batch size
H_ = 3  # Height
W_ = 2  # Width

# Manually create w_idx and h_idx (assuming this is given or can be derived)
w_idx = torch.arange(W_).view(1, 1, W_).expand(B_, H_, W_)  # Shape (B, H, W)
h_idx = torch.arange(H_).view(1, H_, 1).expand(B_, H_, W_)  # Shape (B, H, W)

# Create the batch offset (similar to torch.arange(B).view(-1, 1))
batch_offset = H_ * W_ * torch.arange(B_).view(B_, 1, 1)  # Shape (B, 1, 1)

# Generate the linear indices
linear_idx = w_idx + W_ * h_idx + batch_offset

# Flatten the linear index to check the order
linear_idx_flattened = linear_idx.view(-1)

linear_idx_flattened.view(B_, H_, W_)

# Create a simple 4x3x2 matrix and fill it with numbers for testing
matrix = torch.arange(B_ * H_ * W_).view(B_, H_, W_)

# Print the results
print("w_idx:\n", w_idx)
print("h_idx:\n", h_idx)
print("batch_offset:\n", batch_offset)
print("linear_idx:\n", linear_idx)
print("linear_idx_flattened:\n", linear_idx_flattened)
print("Original matrix:\n", matrix)
print("Flattened matrix with the indices order:\n", matrix.view(-1)[linear_idx_flattened])
