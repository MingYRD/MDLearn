# AlexNet
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

def BN_conv(in_channels, out_channels, kernelSize, stride=1, padding=0):
    layer = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernelSize, stride, padding),
        nn.BatchNorm2d(out_channels, eps=1e-3),
        nn.ReLU(True)
    )
    return layer


class AlexNet(torch.nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            BN_conv(3, 64, 11, 2, 5),
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            BN_conv(64, 128, 5, 1, 2),
            nn.MaxPool2d(2, 2)
        )
        self.layer3 = nn.Sequential(
            BN_conv(128, 256, 3, 1, 1),
            BN_conv(256, 256, 3, 1, 1),
            BN_conv(256, 512, 3, 1, 1),
            nn.MaxPool2d(2, 2)
        )
        self.layer4 = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(2 * 2 * 512, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

class Alexnet_test:

    def __init__(self, lr=0.01, epochs=10):
        self.lr = lr
        self.epochs = epochs
        self.inc = AlexNet(num_classes=10)
        self.old_time = 0
        self.current_time = 0
        self.ek = []
        self.ek_t = []
        self.error = []

        # self.device = torch.device('mps')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device0 = torch.device('cpu')

    def update_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def smooth_step(self, a, b, c, d, x):
        level_s = 0.01
        level_m = 0.1
        level_n = 0.01
        level_r = 0.005
        if x <= a:
            return level_s
        if a < x <= b:
            return (((x - a) / (b - a)) * (level_m - level_s) + level_s)
        if b < x <= c:
            return level_m
        if c < x <= d:
            return level_n
        if d < x:
            return level_r

    def train(self, train_dataloader, test_dataloader):
        self.inc = self.inc.to(self.device)

        opt = torch.optim.SGD(self.inc.parameters(), lr=self.smooth_step(10, 40, 100, 150, 0),
                              momentum=0.9, weight_decay=1e-4)
        loss_fun = torch.nn.CrossEntropyLoss()
        # scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5, last_epoch=-1)
        acc_arr = 0
        for epoch in range(self.epochs):
            loss_train = 0
            loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
            loop.set_description(f'Epoch [{epoch + 1}/{self.epochs}]')
            for index, (img, labels) in loop:
                img = img.to(self.device)
                labels = labels.to(self.device)
                out = self.inc.forward(img)
                loss = loss_fun(out, labels)
                loss_train += loss
                opt.zero_grad()
                loss.backward()
                opt.step()
                # scheduler.step()
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
            curr_lr = self.smooth_step(10, 40, 100, 150, epoch)
            self.update_lr(opt, curr_lr)
            pre_acc = self.predict(test_dataloader)
            # print('Epoch:{} / {}'.format(str(epoch + 1), str(self.epochs)))
            print("Loss:{} ACC:{}".format(loss_train, pre_acc))
            self.error.append(1 - pre_acc)
            if pre_acc > acc_arr:
                acc_arr = pre_acc
                torch.save(self.inc.state_dict(), "Alexnet.pth")
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
        return ans / k

    def get_ek(self):
        return self.ek, self.ek_t

    def get_err(self):
        return self.error