import torch
import torch.nn as nn
import numpy as np
import time
from tqdm import tqdm

from inception_model import Inception, Inception1, Inception2, Inception3, M2_1, M2_2

class inception_base(nn.Module):

    def __init__(self):
        super(inception_base, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 192, kernel_size=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True)

            # nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(True),
            # nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(True),
            # nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(in_channels=64, out_channels=80, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(80),
            # nn.ReLU(True),
            # nn.Conv2d(in_channels=80, out_channels=192, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(192),
            # nn.ReLU(True),


        )
        self.conv2 = nn.Sequential(
            Inception(192, 64, [96, 128], [16, 32], 32),
            Inception(256, 128, [128, 192], [32, 96], 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Inception(480, 192, [96, 208], [16, 48], 64),
            Inception(512, 160, [112, 224], [24, 64], 64),
            Inception(512, 128, [128, 256], [24, 64], 64),
            Inception(512, 112, [144, 288], [32, 64], 64),
            Inception(528, 256, [160, 320], [32, 128], 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Inception(832, 256, [160, 320], [32, 128], 128),
            Inception(832, 384, [192, 384], [48, 128], 128),
            nn.AvgPool2d(4)


        )
        # V3
        # self.conv2 = nn.Sequential(
        #     Inception1(192, 64, [48, 64], [64, 96, 96], 32),
        #     Inception1(256, 64, [48, 64], [64, 96, 96], 64),
        #     Inception1(288, 64, [48, 64], [64, 96, 96], 64),
        #     M2_1(288, 384, 64, 96, 96),
        #
        #     Inception2(768, 192, [128, 128, 192], [128, 128, 128, 128, 192], 192),
        #     Inception2(768, 192, [128, 128, 192], [128, 128, 128, 128, 192], 192),
        #     Inception2(768, 192, [128, 128, 192], [128, 128, 128, 128, 192], 192),
        #     Inception2(768, 192, [128, 128, 192], [128, 128, 128, 128, 192], 192),
        #     # Inception2(528, 256, [160, 320], [32, 128], 128),
        #     M2_2(768, 192, 320, 192, 192, 192, 192),
        #
        #     Inception3(1280, 320, [384, 384, 384], [384, 384, 384, 448], 192),
        #     Inception3(2048, 384, [384, 384, 384], [384, 384, 384, 448], 128),
        #     nn.AvgPool2d(4)
        #
        # )
        self.FC = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(1024, 10)
            # nn.Linear(2048, 10)
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

        opt = torch.optim.SGD(self.inc.parameters(), lr=self.smooth_step(10, 40, 100, 150, 0), momentum=0.9, weight_decay=0.001)
        loss_fun = torch.nn.CrossEntropyLoss()
        # scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5, last_epoch=-1)
        acc_arr = 0
        self.old_time = time.time()
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
                torch.save(self.inc.state_dict(), "inception_v.pth")

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
        # print('ACC:', ans / k)
        return ans / k

    def get_ek(self):
        return self.ek, self.ek_t

    def get_error(self):
        return self.error

