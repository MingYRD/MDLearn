import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
data_transforms ={
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([.5, .5, .5],[.5, .5, .5])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([.5, .5, .5],[.5, .5, .5])
    ])
}
# ImageFolder 通用的加载器
dataset_train = torchvision.datasets.ImageFolder(root='flowers', transform=data_transforms['train'])
dataset_test = torchvision.datasets.ImageFolder(root='flowers', transform=data_transforms['test'])
# 构建可迭代的数据装载器
train_dataloader = DataLoader(dataset=dataset_train, batch_size=4, shuffle=True)
test_dataloader = DataLoader(dataset=dataset_test, batch_size=4, shuffle=False)


cn = 1
for img, labels in train_dataloader:
    print(img.shape)
    img = img[0]  # 若出错可改成img = image[0].cuda 试试
    img = img.numpy()  # FloatTensor转为ndarray
    img = np.transpose(img, (1, 2, 0))  # 把channel那一维放到最后
    # # 显示图片
    # img = img.squeeze()
    plt.imshow(img)
    plt.show()
    cn += 1
    if cn > 3:
        break

