import torch
from vit_base import vit_test
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import os
# from vit_2 import vit_test
cpu_num = 2  # 这里设置成你想运行的CPU个数
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
os.environ["CUDA_VISIBLE_DEVICES"] = '7'
os.environ['CUDA_LAUNCH_BLOCKING'] = '7'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device0 = torch.device('cpu')
transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                      transforms.RandomCrop(32, padding=4),
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
epochs = 200
v = vit_test(lr, epochs)
v.train(train_dataloader, test_dataloader)

ek, ek_t = v.get_ek()
err = v.get_error()
for i in range(len(ek)):
    ek[i] = ek[i].detach().numpy()
    ek_t[i] = ek_t[i].detach().numpy()
# for i in range(len(err)):
#     err[i] = err[i].detach().numpy()
ek = np.array(ek)
ek_t = np.array(ek_t)
err = np.array(err)
np.save('vit3_ek.npy', ek)
np.save('vit3_ekt.npy', ek_t)
np.save('vit3_err.npy', err)

