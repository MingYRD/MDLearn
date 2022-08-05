import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm


class vgg_model(nn.Module):

    def __init__(self, pattern=None, init_flag=False, out_channels=256, end_shape=4, hidden_num=512, out_num=10):
        super(vgg_model, self).__init__()
        self.pattern = pattern
        self.hidden_num = hidden_num
        self.out_num = out_num
        self.end_shape = end_shape
        self.out_channels = out_channels
        self.FC = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.out_channels*self.end_shape*self.end_shape, self.hidden_num),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            # nn.Linear(self.hidden_num, self.hidden_num),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(self.hidden_num, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.out_num),
        )
        # self.FC2 = nn.Softmax(dim=0)
        if init_flag:
            self._init_w()
        self.layers = []
        pre = 3
        for p in self.pattern:
            if p != 'M' and p != 'LRN':
                if p == 255 or p == 511:
                    p = p + 1
                    self.layers += [nn.Conv2d(in_channels=pre, out_channels=p, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(p), nn.ReLU(inplace=True)]

                else:
                    self.layers += [nn.Conv2d(in_channels=pre, out_channels=p, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(p), nn.ReLU(inplace=True)]
                pre = p
            elif p == 'LRN':
                self.layers += [nn.LocalResponseNorm(size=2)]
            else:
                self.layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        self.conv = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.conv(x)
        # x = self.FC(x)
        return self.FC(x)

    def _init_w(self):
        pass

class vgg_test:

    def __init__(self, choice: str, lr: float, epochs: int, dif: int):
        self.lr = lr
        self.error = []
        self.epochs = epochs
        self.choice = choice
        self.dif = dif
        if self.dif == 0:
            self.combination = {
                'A': [64, 'M', 128, 'M', 256, 256, 'M'],
                'A-LRN': [64, 'LRN', 'M', 128, 'M', 256, 256, 'M'],
                'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M'],
                'C': [64, 64, 'M', 128, 128, 'M', 256, 256, 255, 'M'],
                'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M'],
                'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M']
            }
            self.pattern = self.combination[self.choice]
            self.vgg = vgg_model(pattern=self.pattern, init_flag=False)
        else:
            self.combination = {
                'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
                'A-LRN': [64, 'LRN', 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
                'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
                'C': [64, 64, 'M', 128, 128, 'M', 256, 256, 255, 'M', 512, 512, 511, 'M', 512, 512, 511, 'M'],
                'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
                'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,
                      'M']
            }
            self.pattern = self.combination[self.choice]
            self.vgg = vgg_model(pattern=self.pattern, out_channels=512, end_shape=1, hidden_num=512, init_flag=False)

        self.ek = []
        self.ek_t = []
        self.s_ans = None
        self.y_t = None
        self.out_pred = None

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
        self.vgg = self.vgg.to(self.device)

        opt = torch.optim.SGD(self.vgg.parameters(), lr=self.smooth_step(10, 40, 100, 150, 0), momentum=0.9,
                              weight_decay=1e-4)
        # opt = torch.optim.SGD(self.inc.parameters(), lr=self.lr, momentum=0.9,
        #                       weight_decay=1e-4)
        loss_fun = torch.nn.CrossEntropyLoss()
        # scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=40, gamma=0.5, last_epoch=-1)
        acc_arr = 0
        for epoch in range(self.epochs):
            loss_train = 0
            loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
            loop.set_description(f'Epoch [{epoch + 1}/{self.epochs}]')
            for index, (img, labels) in loop:
                img = img.to(self.device)
                labels = labels.to(self.device)
                out = self.vgg.forward(img)
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
                    outputs = self.vgg.forward(img)
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
                torch.save(self.vgg.state_dict(), "vgg_16.pth")

    def predict(self, test_loader):
        ans = 0
        k = 0
        for img, labels in test_loader:
            img = img.to(self.device)
            outputs = self.vgg.forward(img)
            outputs = outputs.to(self.device0)
            self.out_pred = outputs.detach().numpy()
            self.s_ans = np.argmax(outputs.detach().numpy(), axis=1)
            self.y_t = labels.detach().numpy()
            k += self.s_ans.shape[0]
            for j in range(self.s_ans.shape[0]):
                if self.s_ans[j] == self.y_t[j]:
                    ans += 1
        # print('ACC:', ans / k)
        return ans / k

    def get_ek(self):
        return self.ek, self.ek_t

    def get_error(self):
        return self.error