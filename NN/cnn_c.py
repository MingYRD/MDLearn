import torch
import torch.nn as nn
import numpy as np
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from cnn import cnn_lx, cnn_train


train_data = torchvision.datasets.CIFAR10(
    root="data_CIFAR",
    download=True,
    train=True,
    transform=torchvision.transforms.ToTensor()
)
test_data = torchvision.datasets.CIFAR10(
    root="data_CIFAR",
    download=True,
    train=False,
    transform=torchvision.transforms.ToTensor()
)

train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=128, shuffle=False)

