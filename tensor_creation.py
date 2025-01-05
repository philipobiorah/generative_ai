import torch

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a tensor on the appropriate device
tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], device=device)

# Assertions
assert tensor.device.type in {"cuda", "cpu"}
assert tensor.shape == (3, 3)

print(f"Tensor: {tensor}")
print(f"Device: {tensor.device}")