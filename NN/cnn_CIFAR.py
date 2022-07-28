import torch
import torch.nn as nn
import numpy as np
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
from alive_progress import alive_bar
from Inception.inception_model import Inception
"""
data pre
"""

# train_data = torchvision.datasets.MNIST(
#     root="data",
#     download=True,
#     train=True,
#     transform=torchvision.transforms.ToTensor()
# )
# test_data = torchvision.datasets.MNIST(
#     root="data",
#     download=True,
#     train=False,
#     transform=torchvision.transforms.ToTensor()
# )
#
# train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
# test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

# cn = 1
# for img, labels in train_dataloader:
#     img = img[0]  # 若出错可改成img = image[0].cuda 试试
#     img = img.numpy()  # FloatTensor转为ndarray
#     img = np.transpose(img, (1, 2, 0))  # 把channel那一维放到最后
#     # 显示图片
#     plt.imshow(img)
#     plt.show()
#     cn += 1
#     if cn > 6:
#         break
#

"""
data solve
"""


class cnn_lx(torch.nn.Module):

    def __init__(self):
        super(cnn_lx, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16,
                      kernel_size=(3, 3), stride=(1, 1),
                      padding=1),
            # nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 64, 3, 1, 1),
            # nn.BatchNorm2d(64),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, 3, 1, 1),
            # nn.BatchNorm2d(128),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, 3, 1, 1),
            # nn.BatchNorm2d(256),
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(),  # 压缩为1维
            nn.Linear(256 * 2 * 2, 512),
            nn.ReLU(inplace=True),  # 激励函数
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)


class cnn_train:

    def __init__(self, lr=0.1, epochs=10, cnn=None):
        self.lr = lr
        self.epochs = epochs
        self.cnn = cnn
        self.old_time = 0
        self.current_time = 0
        self.ek = []
        self.ek_t = []

        self.device = torch.device('mps')
        self.device0 = torch.device('cpu')

    def train(self, train_dataloader, test_dataloader):

        self.cnn = self.cnn.to(self.device)

        opt = torch.optim.SGD(self.cnn.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.001)
        loss_fun = torch.nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5, last_epoch=-1)

        self.old_time = time.time()
        with alive_bar(self.epochs) as bar:
            for epoch in range(self.epochs):
                loss_train = 0
                for img, labels in train_dataloader:
                    img = img.to(self.device)
                    labels = labels.to(self.device)
                    out = self.cnn.forward(img)
                    loss = loss_fun(out, labels)
                    loss_train += loss
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                scheduler.step()
                total_loss = 0  # 保存这次测试总的loss
                with torch.no_grad():  # 下面不需要反向传播，所以不需要自动求导
                    for img, labels in test_dataloader:
                        img = img.to(self.device)
                        labels = labels.to(self.device)
                        outputs = self.cnn.forward(img)
                        loss = loss_fun(outputs, labels)
                        total_loss += loss  # 累计误差
                self.ek.append(loss_train.to(self.device0))
                self.ek_t.append(total_loss.to(self.device0))
                bar()
                print('Epoch:{} / {}'.format(str(epoch + 1), str(self.epochs)))
                print("第{}次训练的Loss:{}".format(epoch + 1, total_loss))
                self.predict(test_dataloader)
        self.current_time = time.time()
        # print('Time:' + str(self.current_time - self.old_time) + 's')
        # torch.save(self.cnn, "cnn_digit.nn")

    def predict(self, test_dataloader):
        ans = 0
        k = 0
        for img, labels in test_dataloader:
            img = img.to(self.device)
            outputs = self.cnn.forward(img)
            outputs = outputs.to(self.device0)
            s_ans = np.argmax(outputs.detach().numpy(), axis=1)
            y_t = labels.detach().numpy()
            k += s_ans.shape[0]
            for j in range(s_ans.shape[0]):
                if s_ans[j] == y_t[j]:
                    ans += 1
        print('ACC:', ans / k)

    def get_ek(self):
        return self.ek, self.ek_t











