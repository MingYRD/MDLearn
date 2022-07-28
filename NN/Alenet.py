# AlexNet
import time

import numpy as np
import torch
from alive_progress import alive_bar


class AlexNet(torch.nn.Module):
    def __init__(self, num_classes, init_weights=False):
        super(AlexNet, self).__init__()
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=2),
                                          # torch.nn.BatchNorm2d(64),
                                          torch.nn.ReLU(inplace=True),
                                          torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0))

        self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(64, 192, kernel_size=4, stride=1, padding=1),
                                          # torch.nn.BatchNorm2d(192),
                                          torch.nn.ReLU(inplace=True),
                                          torch.nn.MaxPool2d(kernel_size=2, stride=1, padding=0))

        self.layer3 = torch.nn.Sequential(torch.nn.Conv2d(192, 384, kernel_size=3, padding=1),
                                          # torch.nn.BatchNorm2d(384),
                                          torch.nn.ReLU(inplace=True))

        self.layer4 = torch.nn.Sequential(torch.nn.Conv2d(384, 256, kernel_size=3, padding=1),
                                          # torch.nn.BatchNorm2d(256),
                                          torch.nn.ReLU(inplace=True))

        self.layer5 = torch.nn.Sequential(torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                          # torch.nn.BatchNorm2d(256),
                                          torch.nn.ReLU(inplace=True),
                                          torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # self.avgpool = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(output_size=(3, 3)))

        self.fc1 = torch.nn.Sequential(torch.nn.Dropout(p=0.5, inplace=False),
                                       torch.nn.Linear(256 * 3 * 3, 1024),
                                       torch.nn.ReLU(inplace=True))

        self.fc2 = torch.nn.Sequential(torch.nn.Dropout(p=0.5, inplace=False),
                                       torch.nn.Linear(1024, 1024),
                                       torch.nn.ReLU(inplace=True))

        self.fc3 = torch.nn.Sequential(torch.nn.Dropout(p=0.5, inplace=False),
                                       torch.nn.Linear(1024, num_classes))
        if init_weights:
            pass

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        # out = self.avgpool(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)

class Alexnet_test:

    def __init__(self, lr=0.01, epochs=10):
        self.lr = lr
        self.epochs = epochs
        self.inc = AlexNet(init_weights=False, num_classes=10)
        self.old_time = 0
        self.current_time = 0
        self.ek = []
        self.ek_t = []

        # self.device = torch.device('mps')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
