import torch
import torch.nn.functional as F

def BinaryCrossEntropy(y: torch.Tensor, y_hat: torch.Tensor) -> float:
    return F.binary_cross_entropy(y, y_hat)
