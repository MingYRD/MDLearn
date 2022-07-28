# 使用torch.nn包来构建神经网络.
import numpy as np
import torch.nn as nn
import torch
import time
import torch.nn.functional as F

from alive_progress import alive_bar


class LeNet(nn.Module): 					# 继承于nn.Module这个父类
    def __init__(self):						# 初始化网络结构
        super(LeNet, self).__init__()    	# 多继承需用到super函数
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):			 # 正向传播过程
        x = F.relu(self.conv1(x))    # input(3, 32, 32) output(16, 28, 28)
        x = self.pool1(x)            # output(16, 14, 14)
        x = F.relu(self.conv2(x))    # output(32, 10, 10)
        x = self.pool2(x)            # output(32, 5, 5)
        x = x.view(-1, 32*5*5)       # output(32*5*5)
        x = F.relu(self.fc1(x))      # output(120)
        x = F.relu(self.fc2(x))      # output(84)
        x = self.fc3(x)              # output(10)
        return x


class Lenet_test:

    def __init__(self, lr=0.01, epochs=10):
        self.lr = lr
        self.epochs = epochs
        self.inc = LeNet()
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
