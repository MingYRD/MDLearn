import torch.nn as nn
import torch

class resnet_block(nn.Module):

    def __init__(self, input_channels, num_channels, s=1, is1x1=False):
        super(resnet_block, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, stride=s, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)

        if is1x1:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=s)
        else:
            self.conv3 = None

        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.relu(y)
