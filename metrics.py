import torch

def get_accuracy(predict: torch.Tensor, target: torch.Tensor) -> float:
    n = predict.shape[0]
    return torch.sum((predict >= 0.5) == (target >= 0.5)).item() / n
