import torch
import torch.nn as nn
import numpy as np
import time
from alive_progress import alive_bar
from ResNet_block import resnet_block

class resnet_base(nn.Module):

    def __init__(self):
        super(resnet_base, self).__init__()
        self.conv1 = nn.Sequential(
            # nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv2 = nn.Sequential(*self.make_resnet_block(64, 64, 2, first_block=True))
        self.conv3 = nn.Sequential(*self.make_resnet_block(64, 128, 2))
        self.conv4 = nn.Sequential(*self.make_resnet_block(128, 256, 2))
        self.conv5 = nn.Sequential(*self.make_resnet_block(256, 512, 2))

        self.FC = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 10)
        )

    def make_resnet_block(self, input_channels, num_channels, num_residuals, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(resnet_block(input_channels, num_channels, is1x1=True, s=2))
            else:
                blk.append(resnet_block(num_channels, num_channels))
        return blk

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return self.FC(x)

class resnet_test:

    def __init__(self, lr=0.01, epochs=10):
        self.lr = lr
        self.epochs = epochs
        self.inc = resnet_base()
        self.old_time = 0
        self.current_time = 0
        self.ek = []
        self.ek_t = []

        self.device = torch.device('mps')
        self.device0 = torch.device('cpu')

    def train(self, train_dataloader, test_dataloader):
        self.inc = self.inc.to(self.device)

        opt = torch.optim.SGD(self.inc.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
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


