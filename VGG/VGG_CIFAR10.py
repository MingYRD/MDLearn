import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from VGG_model import vgg_test

transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

train_data = torchvision.datasets.CIFAR10(
    root="../NN/data_CIFAR",
    download=True,
    train=True,
    transform=transform_train
)
test_data = torchvision.datasets.CIFAR10(
    root="../NN/data_CIFAR",
    download=True,
    train=False,
    transform=transform
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
#
# classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# def image_show(img):
#     img = img / 2 + 0.5
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
# def label_show(loader):
#     global classes
#     dataiter = iter(loader)  # 迭代遍历图片
#     images, labels = dataiter.next()
#     image_show(make_grid(images))
#     print(' '.join('%5s' % classes[labels[j]] for j in range(128)))
#     return images,labels
# #label_show(train_loader)

lr = 0.01
epochs = 50
choice = 'B'
vgg = vgg_test(choice, lr, epochs, 1)

vgg.train(train_dataloader, test_dataloader)
ek, ek_t = vgg.get_ek()

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
