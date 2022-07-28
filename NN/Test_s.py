import numpy as np
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from Lenet import Lenet_test
from Alenet import Alexnet_test


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
lr = 0.01
epochs = 50
print("AlexNet")
inc = Alexnet_test(lr, epochs)
inc.train(train_dataloader, test_dataloader)

print("LeNet")
inc1 = Lenet_test(lr, epochs)
inc1.train(train_dataloader, test_dataloader)
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
