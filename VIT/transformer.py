import torch
import torch.nn as nn

class transformer(nn.Module):

    def __init__(self):
        super(transformer, self).__init__()
        self.en = nn.MultiheadAttention(32, 1)
        self.c = nn.Sequential(
            nn.Embedding(8, 7),
        )

