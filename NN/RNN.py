import torch
import torch.nn as nn

class RNN_model(nn.Module):

    def __init__(self):
        super(RNN_model, self).__init__()
        self.r1 = nn.LSTM(
            input_size=28,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        r = self.r1(x, None)
        return self.out(r[:, -1, :])

