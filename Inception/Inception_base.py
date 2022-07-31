import torch
import torch.nn as nn
import numpy as np
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import time
from alive_progress import alive_bar
from inception_model import Inception

class inception_base(nn.Module):

    def __init__(self):
        super(inception_base, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            Inception(64, 32, [48, 64], [8, 8, 16], 16),
            Inception(128, 64, [96, 128], [16, 16, 32], 32),
            Inception(256, 64, [96, 128], [16, 16, 32], 32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Inception(256, 64, [96, 128], [16, 16, 32], 32),
            Inception(256, 128, [192, 256], [32, 32, 64], 64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.FC = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*4*4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.FC(x)

class Inception_test:

    def __init__(self, lr=0.01, epochs=10):
        self.lr = lr
        self.epochs = epochs
        self.inc = inception_base()
        self.old_time = 0
        self.current_time = 0
        self.ek = []
        self.ek_t = []

        self.device = torch.device('mps')
        self.device0 = torch.device('cpu')

    def train(self, train_dataloader, test_dataloader):
        self.inc = self.inc.to(self.device)

        opt = torch.optim.SGD(self.inc.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.001)
        loss_fun = torch.nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5, last_epoch=-1)

        self.old_time = time.time()
        with alive_bar(self.epochs) as bar:
            for epoch in range(self.epochs):
                loss_train = 0
                for img, labels in train_dataloader:
                    img = img.to(self.device)
                    labels = labels.to(self.device)
                    out = self.inc.forward(img)
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
                        outputs = self.inc.forward(img)
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
            outputs = self.inc.forward(img)
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


