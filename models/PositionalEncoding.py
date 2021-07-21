import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(PositionalEncoding, self).__init__()
        self.position_embeddings = nn.Parameter(torch.zeros(1, seq_length, embedding_dim)) #8

    def forward(self, x, position_ids=None):
        position_embeddings = self.position_embeddings
        return x + position_embeddings
