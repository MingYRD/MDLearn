import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from VGG_model import vgg_test

import os

cpu_num = 2  # 这里设置成你想运行的CPU个数
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
os.environ["CUDA_VISIBLE_DEVICES"] = '7'
os.environ['CUDA_LAUNCH_BLOCKING'] = '7'
transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                      transforms.RandomCrop(32, padding=4),
                                      transforms.ColorJitter(0.5, 0.5, 0.5),  # 颜色变换
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
                                     )
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
test_dataloader = DataLoader(test_data, batch_size=256, shuffle=False)


lr = 0.1
epochs = 220
version = 'D'
inc = vgg_test(version, lr, epochs, 1)
inc.train(train_dataloader, test_dataloader)

err = inc.get_error()
err = np.array(err)
np.save('err_vgg16.npy', err)
#
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

