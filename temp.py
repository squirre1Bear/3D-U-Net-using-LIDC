import torch
import numpy as np

target = torch.tensor([[[[0, 1, 0], [0, 1, 1]]], [[[1, 0, 1], [0, 0, 0]]]])
print(target)  # Output: torch.Size([2, 1, 2, 3])


target = target.squeeze(1)
print(target)  # Output: torch.Size([2, 2, 3])