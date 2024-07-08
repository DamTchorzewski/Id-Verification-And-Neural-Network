# Zrozumienie Tensorów w PyTorch:
# Utwórz tensor w PyTorch i wykonaj na nim podstawowe operacje
# (dodawanie, mnożenie, transpozycja)

import torch

print("Tensor Basics")

# Create a tensor
tensor1 = torch.tensor([[1, 2], [3, 4]])
print("Tensor 1:")
print(tensor1)

# Create another tensor
tensor2 = torch.tensor([[5, 6], [7, 8]])
print("\nTensor 2:")
print(tensor2)

# Add the tensors
tensor3 = tensor1 + tensor2
print("\nTensor 1 + Tensor 2:")
print(tensor3)

# Multiply the tensors
tensor4 = tensor1 * tensor2
print("\nTensor 1 * Tensor 2:")
print(tensor4)

# Transpose the first tensor
tensor5 = tensor1.t()
print("\nTranspose of Tensor 1:")
print(tensor5)