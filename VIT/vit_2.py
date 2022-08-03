import time
import numpy as np
import torch
from alive_progress import alive_bar
from vit_pytorch import ViT
# from vit_start import ViT

class vit_test:

    def __init__(self, lr=0.01, epochs=10):
        self.lr = lr
        self.epochs = epochs
        self.v = torch.load('vit.pth')
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
        # self.v = self.v.to(self.device)

        opt = torch.optim.SGD(self.v.parameters(), lr=self.smooth_step(10, 40, 100, 150, 0), momentum=0.9,
                              weight_decay=1e-4)
        loss_fun = torch.nn.CrossEntropyLoss()
        # scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5, last_epoch=-1)
        acc_arr = 0
        self.old_time = time.time()
        with alive_bar(self.epochs) as bar:
            for epoch in range(self.epochs):
                loss_train = 0
                for img, labels in train_dataloader:
                    img = img.to(self.device)
                    labels = labels.to(self.device)
                    out = self.v(img)
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
                        outputs = self.v(img)
                        loss = loss_fun(outputs, labels)
                        total_loss += loss  # 累计误差
                self.ek.append(loss_train.to(self.device0))
                self.ek_t.append(total_loss.to(self.device0))
                curr_lr = self.smooth_step(10, 40, 100, 150, epoch)
                self.update_lr(opt, curr_lr)
                bar()
                print('Epoch:{} / {}'.format(str(epoch + 1), str(self.epochs)))
                print("第{}次训练的Loss:{}".format(epoch + 1, loss_train))
                pre_acc = self.predict(test_dataloader)
                self.error.append(1 - pre_acc)
                if pre_acc > acc_arr:
                    acc_arr = pre_acc
                    torch.save(self.v.state_dict(), "vit_t1.pth")
        self.current_time = time.time()
        # print('Time:' + str(self.current_time - self.old_time) + 's')
        # torch.save(self.cnn, "cnn_digit.nn")

    def predict(self, test_dataloader):
        ans = 0
        k = 0
        for img, labels in test_dataloader:
            img = img.to(self.device)
            outputs = self.v.forward(img)
            outputs = outputs.to(self.device0)
            s_ans = np.argmax(outputs.detach().numpy(), axis=1)
            y_t = labels.detach().numpy()
            k += s_ans.shape[0]
            for j in range(s_ans.shape[0]):
                if s_ans[j] == y_t[j]:
                    ans += 1
        print('ACC:', ans / k)
        return ans / k

    def get_ek(self):
        return self.ek, self.ek_t

    def get_error(self):
        return self.error
