import torch
import torch.nn as nn
import numpy as np
import time

from tqdm import tqdm

from ResNet_block import resnet_block, resnet_block_d



class resnet_base(nn.Module):

    def __init__(self):
        super(resnet_base, self).__init__()
        self.conv1 = nn.Sequential(
            # nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            # nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # resnet-18
        self.conv2 = nn.Sequential(*self.make_resnet_block(64, 64, 2, first_block=True))
        self.conv3 = nn.Sequential(*self.make_resnet_block(64, 128, 2))
        self.conv4 = nn.Sequential(*self.make_resnet_block(128, 256, 2))
        self.conv5 = nn.Sequential(*self.make_resnet_block(256, 512, 2))
        # resnet - 34
        # self.conv2 = nn.Sequential(*self.make_resnet_block(64, 64, 3, first_block=True))
        # self.conv3 = nn.Sequential(*self.make_resnet_block(64, 128, 4))
        # self.conv4 = nn.Sequential(*self.make_resnet_block(128, 256, 6))
        # self.conv5 = nn.Sequential(*self.make_resnet_block(256, 512, 3))

        # resnet - 20
        # self.conv2 = nn.Sequential(*self.make_resnet_block(64, 64, 3, first_block=True))
        # self.conv3 = nn.Sequential(*self.make_resnet_block(64, 128, 3))
        # self.conv4 = nn.Sequential(*self.make_resnet_block(128, 256, 3))

        # self.conv2 = nn.Sequential(*self.make_resnet_block(16, 16, 3, first_block=True))
        # self.conv3 = nn.Sequential(*self.make_resnet_block(16, 32, 3))
        # self.conv4 = nn.Sequential(*self.make_resnet_block(32, 64, 3))

        # resnet - 32
        # self.conv2 = nn.Sequential(*self.make_resnet_block(64, 64, 5, first_block=True))
        # self.conv3 = nn.Sequential(*self.make_resnet_block(64, 128, 5))
        # self.conv4 = nn.Sequential(*self.make_resnet_block(128, 256, 5))
        # self.conv5 = nn.Sequential(*self.make_resnet_block(256, 512, 4))

        # resnet - 50
        # self.conv2 = nn.Sequential(*self.make_resnet_block(64, 64, 3, first_block=True, out_channels=32))
        # self.conv3 = nn.Sequential(*self.make_resnet_block(64, 128, 4, out_channels=96))
        # self.conv4 = nn.Sequential(*self.make_resnet_block(128, 256, 6, out_channels=192))
        # self.conv5 = nn.Sequential(*self.make_resnet_block(256, 512, 3, out_channels=384))

        # resnet - 101
        # self.conv2 = nn.Sequential(*self.make_resnet_block(64, 64, 3, first_block=True, out_channels=32))
        # self.conv3 = nn.Sequential(*self.make_resnet_block(64, 128, 4, out_channels=96))
        # self.conv4 = nn.Sequential(*self.make_resnet_block(128, 256, 23, out_channels=192))
        # self.conv5 = nn.Sequential(*self.make_resnet_block(256, 512, 3, out_channels=384))

        self.FC = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 10)
            # nn.Linear(256, 10)
            # nn.Linear(64, 10)
        )

    def make_resnet_block(self, input_channels, num_channels, num_residuals, first_block=False, out_channels=64):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(resnet_block(input_channels, num_channels, is1x1=True, s=2))
                # blk.append(resnet_block_d(input_channels, num_channels, is1x1=True, s=2, out_channels=out_channels))
            else:
                blk.append(resnet_block(num_channels, num_channels))
                # blk.append(resnet_block_d(num_channels, num_channels, out_channels=out_channels))
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
        # resnet = torch.load('inc_resnet_18.pth', map_location=torch.device('cpu'))
        # self.inc.load_state_dict(resnet)
        self.old_time = 0
        self.current_time = 0
        self.ek = []
        self.ek_t = []
        self.error = []

        # self.device = torch.device('mps')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device0 = torch.device('cpu')

    # In[1] 设置一个通过优化器更新学习率的函数
    def update_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def smooth_step(self, a, b, c, d, e, x):
        level_s = 0.01
        level_m = 0.1
        level_n = 0.01
        level_r = 0.005
        level_r2 = 0.0001
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
        if x > e:
            return level_r2

    def train(self, train_dataloader, test_dataloader):
        self.inc = self.inc.to(self.device)

        opt = torch.optim.SGD(self.inc.parameters(), lr=self.smooth_step(10, 40, 100, 150, 200, 0), momentum=0.9,
                              weight_decay=1e-4)
        # opt = torch.optim.SGD(self.inc.parameters(), lr=self.lr, momentum=0.9,
        #                       weight_decay=1e-4)
        loss_fun = torch.nn.CrossEntropyLoss()
        # scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=40, gamma=0.5, last_epoch=-1)
        acc_arr = 0
        self.old_time = time.time()
        for epoch in range(self.epochs):
            loss_train = 0
            loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
            loop.set_description(f'Epoch [{epoch + 1}/{self.epochs}]')

            self.inc.train()
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
            pre_acc = self.predict(test_dataloader)
            # total_loss = 0  # 保存这次测试总的loss
            # with torch.no_grad():  # 下面不需要反向传播，所以不需要自动求导
            #     for img, labels in test_dataloader:
            #         img = img.to(self.device)
            #         labels = labels.to(self.device)
            #         outputs = self.inc.forward(img)
            #         loss = loss_fun(outputs, labels)
            #         total_loss += loss  # 累计误差
            # self.ek.append(loss_train.to(self.device0))
            # self.ek_t.append(total_loss.to(self.device0))
            curr_lr = self.smooth_step(10, 40, 100, 150, 200, epoch)
            self.update_lr(opt, curr_lr)
            # print('Epoch:{} / {}'.format(str(epoch + 1), str(self.epochs)))
            print("Loss:{} ACC:{}".format(loss_train, pre_acc))
            self.error.append(1 - pre_acc)
            if pre_acc > acc_arr:
                acc_arr = pre_acc
                torch.save(self.inc.state_dict(), "inc_resnet_18.pth")

        # print('Time:' + str(self.current_time - self.old_time) + 's')
        # torch.save(self.cnn, "cnn_digit.nn")

    def predict(self, test_dataloader):
        ans = 0
        k = 0
        self.inc.eval()
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
