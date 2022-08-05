import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

class BP_Re(torch.nn.Module):
    def __init__(self):
        super(BP_Re, self).__init__()
        self.fc1 = torch.nn.Linear(1, 32)
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

    def predict(self, x):
        y_pred = self.forward(x)
        return y_pred


data_train_a = pd.read_csv('../other/sinc_train.txt', sep=' ', header=None, names=['B', 'A'])
data_test_a = pd.read_csv('../other/sinc_test.txt', sep=' ', header=None, names=['B', 'A'])
cols = data_train_a.shape[1]
X_train = data_train_a.iloc[:, 1:cols]
y_train = data_train_a.iloc[:, 0:1]
X_train = np.asarray(X_train.values)
y_train = np.asarray(y_train.values)
cols = data_test_a.shape[1]
X_test = data_test_a.iloc[:, 1:cols]
y_test = data_test_a.iloc[:, 0:1]
X_test = np.asarray(X_test.values)
y_test = np.asarray(y_test.values)

maxy = np.max(y_train) + 0.2
miny = np.min(y_train) - 0.2

y_train = y_train / (maxy - miny) - miny / (maxy - miny)
y_test = y_test / (maxy - miny) - miny / (maxy - miny)


x_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)

x_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)
# device = torch.device('mps')
# device0 = torch.device('cpu')

# x_train = x_train.to(device)
# y_train = y_train.to(device)
# x_train = x_train.cuda
bp = BP_Re()

epochs = 25000
lr = 0.001
loss_fun = torch.nn.MSELoss(reduction='sum')
opt = torch.optim.SGD(bp.parameters(), lr=lr)
losses = []


for epoch in range(epochs):
    print(epoch)
    y_pred = bp.forward(x_train)
    loss = loss_fun(y_pred, y_train)
    # losses.append(loss)
    opt.zero_grad()
    loss.backward()
    opt.step()

yk = bp.forward(x_test)
# print(yk)
fig, ax = plt.subplots(figsize=(12, 8))  # 分解为两个元组对象
ax.plot(X_test, yk.detach().numpy(), 'r', label='p')  # xy，颜色，标签
ax.scatter(X_test, y_test, label='Test Data')   # 散点图
ax.legend(loc=2)   # 控制图例的位置为第二象限
ax.set_xlabel('A')
ax.set_ylabel('B')
ax.set_title('BP - Linear')
plt.show()

