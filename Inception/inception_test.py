import numpy as np
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from Inception_base import Inception_test


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
inc = Inception_test(lr, epochs)
inc.train(train_dataloader, test_dataloader)
err = inc.get_error()
err = np.array(err)
np.save('err_inceptionv3.npy', err)
# ek, ek_t = inc.get_ek()
#
# for i in range(len(ek)):
#     ek[i] = ek[i].detach().numpy()
#     ek_t[i] = ek_t[i].detach().numpy()
# x = np.linspace(0, epochs, num=epochs)
# fig, ax = plt.subplots(figsize=(12, 8))
# ax.plot(x, ek, 'b', label='train')
# ax.plot(x, ek_t, 'g', label='test')
# ax.legend(loc=2)
# ax.set_xlabel('Iter Number')
# ax.set_ylabel('Ek')
# plt.show()
