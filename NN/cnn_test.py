import torch
import torch.nn as nn
import numpy as np
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from cnn import cnn_lx, cnn_train
"""
data pre
"""

train_data = torchvision.datasets.MNIST(
    root="data",
    download=True,
    train=True,
    transform=torchvision.transforms.ToTensor()
)
test_data = torchvision.datasets.MNIST(
    root="data",
    download=True,
    train=False,
    transform=torchvision.transforms.ToTensor()
)

train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)


lr = 0.1
epochs = 10
cnn_m = cnn_lx()
cnn = cnn_train(lr, epochs, cnn_m)

cnn.train(train_dataloader, test_dataloader)
cnn.predict(train_dataloader)
cnn.predict(test_dataloader)
ek, ek_t = cnn.get_ek()
for i in range(len(ek)):
    ek[i] = ek[i].detach().numpy()
    ek_t[i] = ek_t[i].detach().numpy()

x = np.linspace(0, 10, num=10)
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x, ek, 'b', label='train')
ax.plot(x, ek_t, 'g', label='test')
ax.legend(loc=2)
ax.set_xlabel('Iter Number')
ax.set_ylabel('Ek')
plt.show()
