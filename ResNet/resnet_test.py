import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from Resnet_base import resnet_test

import os

cpu_num = 2  # 这里设置成你想运行的CPU个数
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
os.environ['CUDA_LAUNCH_BLOCKING'] = '2'
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
epochs = 230
inc = resnet_test(lr, epochs)
inc.train(train_dataloader, test_dataloader)

# ek, ek_t = inc.get_ek()
err = inc.get_error()
# for i in range(len(ek)):
#     ek[i] = ek[i].detach().numpy()
#     ek_t[i] = ek_t[i].detach().numpy()
# # for i in range(len(err)):
# #     err[i] = err[i].detach().numpy()
# ek = np.array(ek)
# ek_t = np.array(ek_t)
err = np.array(err)
# np.save('ek_18.npy', ek)
# np.save('ek_t_18.npy', ek_t)
np.save('err_18.npy', err)
#


# x = np.linspace(0, epochs, num=epochs)
# fig, ax = plt.subplots(figsize=(12, 8))
# ax.plot(x, ek, 'b', label='train')
# ax.plot(x, ek_t, 'g', label='test')
# ax.legend(loc=2)
# ax.set_xlabel('Iter Number')
# ax.set_ylabel('Ek')
# plt.show()
