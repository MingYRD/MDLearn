import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from VGG_model import vgg_test


# train_data = torchvision.datasets.STL10(
#     root="data_STL",
#     download=True,
#     split='train',
#     transform=torchvision.transforms.ToTensor()
# )
# test_data = torchvision.datasets.STL10(
#     root="data_STL",
#     download=True,
#     split='test',
#     transform=torchvision.transforms.ToTensor()
# )
#
#
# batch = 256
# train_dataloader = DataLoader(train_data, batch_size=batch, shuffle=True)
# test_dataloader = DataLoader(test_data, batch_size=batch, shuffle=False)

data_transforms ={
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([.5, .5, .5], [.5, .5, .5])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([.5, .5, .5], [.5, .5, .5])
    ])
}
# ImageFolder 通用的加载器
dataset_train = torchvision.datasets.ImageFolder(root='flowers', transform=data_transforms['train'])
dataset_test = torchvision.datasets.ImageFolder(root='flowers', transform=data_transforms['test'])
# 构建可迭代的数据装载器
train_dataloader = DataLoader(dataset=dataset_train, batch_size=16, shuffle=True)
test_dataloader = DataLoader(dataset=dataset_test, batch_size=16, shuffle=False)

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

lr = 0.01
epochs = 10

vgg = vgg_test(choice='D', lr=lr, epochs=epochs, dif=1)
vgg.train(train_dataloader, test_dataloader)

