import torch
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

def data_set_laryngeal():
    data_transforms ={
        'train': transforms.Compose([
            transforms.RandomResizedCrop(112),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([.5, .5, .5],[.5, .5, .5])
        ]),
        'val': transforms.Compose([
            # transforms.Resize(120),
            transforms.CenterCrop(112),
            transforms.ToTensor(),
            transforms.Normalize([.5, .5, .5],[.5, .5, .5])
        ])
    }

    # ImageFolder 通用的加载器
    dataset_train1 = torchvision.datasets.ImageFolder(root='laryngeal_dataset/FOLD 1', transform=data_transforms['train'])
    dataset_train2 = torchvision.datasets.ImageFolder(root='laryngeal_dataset/FOLD 2', transform=data_transforms['train'])
    dataset_test = torchvision.datasets.ImageFolder(root='laryngeal_dataset/FOLD 3', transform=data_transforms['val'])
    dataset_train = torch.utils.data.ConcatDataset([dataset_train1, dataset_train2])
    # 构建可迭代的数据装载器
    train_dataloader = DataLoader(dataset=dataset_train, batch_size=120, shuffle=True)
    test_dataloader = DataLoader(dataset=dataset_test, batch_size=120, shuffle=False)

    return train_dataloader, test_dataloader

# cn = 0
# for img, labels in train_dataloader:
#     print(img.shape)
#     print(labels)
#     img = img[0]  # 若出错可改成img = image[0].cuda 试试
#     img = img.numpy()  # FloatTensor转为ndarray
#     img = np.transpose(img, (1, 2, 0))  # 把channel那一维放到最后
#     # # 显示图片
#     # img = img.squeeze()
#     plt.imshow(img)
#     plt.show()
#     cn += 1
#     if cn > 4:
#         break



