import torch
import torch.nn as nn
import numpy as np
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import time

"""
data pre
"""

train_data = torchvision.datasets.MNIST(
    root="data",
    download=True,
    train=True,
    transform=torchvision.transforms.ToTensor()
)
test_data = torchvision.datasets.MNIST(
    root="data",
    download=True,
    train=False,
    transform=torchvision.transforms.ToTensor()
)

train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

cn = 1
for img, labels in train_dataloader:
    img = img[0]  # 若出错可改成img = image[0].cuda 试试
    img = img.numpy()  # FloatTensor转为ndarray
    img = np.transpose(img, (1, 2, 0))  # 把channel那一维放到最后
    # 显示图片
    plt.imshow(img)
    plt.show()
    cn += 1
    if cn > 6:
        break


"""
data solve
"""
class cnn_lx(torch.nn.Module):

    def __init__(self):
        super(cnn_lx, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16,
                      kernel_size=(3, 3), stride=(1, 1),
                      padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, 3, 1, 1),  # 第二次卷积，尺寸为32×14×14
            nn.MaxPool2d(2, stride=2),  # 第二次池化，尺寸为32×7×7
            nn.Flatten(),  # 压缩为1维
            nn.Linear(32 * 7 * 7, 16),
            nn.ReLU(),  # 激励函数
            nn.Linear(16, 10)
        )

    def forward(self, x):
        return self.net(x)
    # def predict(self):
    #     pass


lr = 0.1
epochs = 10

cnn = cnn_lx()

opt = torch.optim.SGD(cnn.parameters(), lr=lr)
loss_fun = torch.nn.CrossEntropyLoss()


old_time = time.time()
for epoch in range(epochs):

    for img, labels in train_dataloader:
        out = cnn.forward(img)
        loss = loss_fun(out, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
    total_loss = 0  # 保存这次测试总的loss
    with torch.no_grad():  # 下面不需要反向传播，所以不需要自动求导
        for img, labels in test_dataloader:
            outputs = cnn.forward(img)
            loss = loss_fun(outputs, labels)
            total_loss += loss  # 累计误差
    print("第{}次训练的Loss:{}".format(epoch + 1, total_loss))
current_time = time.time()

torch.save(cnn, "cnn_digit.nn")
ans = 0
k = 0
for img, labels in test_dataloader:
    outputs = cnn.forward(img)
    s_ans = np.argmax(outputs.detach().numpy(), axis=1)
    y_t = labels.detach().numpy()
    k += s_ans.shape[0]
    for j in range(s_ans.shape[0]):
        if s_ans[j] == y_t[j]:
            ans += 1
print('ACC', ans / k)
print('Time'+str(current_time - old_time)+'s')
