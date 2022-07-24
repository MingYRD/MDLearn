import torch
import torch.nn as nn

class RNN_model(nn.Module):

    def __init__(self):
        super(RNN_model, self).__init__()
        self.r1 = nn.RNN()
        self.f = nn.Linear()