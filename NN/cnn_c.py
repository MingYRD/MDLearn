import torch
import torch.nn as nn
import numpy as np
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from cnn_CIFAR import cnn_lx, cnn_train


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

# cn = 1
# for img, labels in train_dataloader:
#     print(img.shape)
#     img = img[0]  # 若出错可改成img = image[0].cuda 试试
#     img = img.numpy()  # FloatTensor转为ndarray
#     img = np.transpose(img, (1, 2, 0))  # 把channel那一维放到最后
#     # # 显示图片
#     # img = img.squeeze()
#     plt.imshow(img)
#     plt.show()
#     cn += 1
#     if cn > 3:
#         break

lr = 0.01
epochs = 50
cnn_m = cnn_lx()
cnn = cnn_train(lr, epochs, cnn_m)
cnn.train(train_dataloader, test_dataloader)
cnn.predict(train_dataloader)
# cnn.predict(test_dataloader)
ek, ek_t = cnn.get_ek()
for i in range(len(ek)):
    ek[i] = ek[i].detach().numpy()
    ek_t[i] = ek_t[i].detach().numpy()
x = np.linspace(0, epochs, num=epochs)
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x, ek, 'b', label='train')
ax.plot(x, ek_t, 'g', label='test')
ax.legend(loc=2)
ax.set_xlabel('Iter Number')
ax.set_ylabel('Ek')
plt.show()