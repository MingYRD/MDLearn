import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from BP_Martrix import BP_M3
from BP_Matrix_General import BP_M

def normalize_f(data):
    return (data - data.min())/(data.max() - data.min())
#
data_train_a = pd.read_csv('../sinc_train.txt', sep=' ', header=None, names=['B', 'A'])
data_test_a = pd.read_csv('../sinc_test.txt', sep=' ', header=None, names=['B', 'A'])
# data_train_a = normalize_f(data_train_a)
# data_test_a = normalize_f(data_test_a)
cols = data_train_a.shape[1]
X_train = data_train_a.iloc[:, 1:cols]
y_train = data_train_a.iloc[:, 0:1]
# y_train = normalize_f(y_train)
X_train = np.asarray(X_train.values)
y_train = np.asarray(y_train.values)
#
# # test
cols = data_test_a.shape[1]
X_test = data_test_a.iloc[:, 1:cols]
y_test = data_test_a.iloc[:, 0:1]
# y_test = normalize_f(y_test)
X_test = np.asarray(X_test.values)
y_test = np.asarray(y_test.values)

maxy = np.max(y_train) + 0.2
miny = np.min(y_train) - 0.2

y_train = y_train / (maxy - miny) - miny / (maxy - miny)
y_test = y_test / (maxy - miny) - miny / (maxy - miny)



it = 31000
eta = 0.001
every_num = []

every_num.append(32)
every_num.append(16)
every_num.append(8)
every_num.append(1)
bpc = BP_M(eta=eta, max_iter=it, every_num=every_num, hide_num=3)
# bpc = BP_M3(eta=0.0015, max_it1er=it, s_num=16)
bpc.init_params(X_train)
bpc.BP_train(X_train, y_train, X_test, y_test)
ek, ek_t = bpc.get_ek()
yk = bpc.predict(X_test)
yy = bpc.predict(X_train)
# maxy = np.max(yk) + 0.2
# miny = np.min(yk) - 0.2
# yk = yk / (maxy - miny) - miny / (maxy - miny)

# x_p = np.arange(-10, 10, 0.004)
fig, ax = plt.subplots(figsize=(12, 8))  # 分解为两个元组对象
ax.plot(X_test, yk, 'r', label='p')  # xy，颜色，标签
ax.scatter(X_test, y_test, label='Test Data')   # 散点图
ax.legend(loc=2)   # 控制图例的位置为第二象限
ax.set_xlabel('A')
ax.set_ylabel('B')
ax.set_title('BP - Linear')
plt.show()



x = np.linspace(0, it, num=it)
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x, ek, 'b', label='train')
ax.plot(x, ek_t, 'g', label='test')
ax.legend(loc=2)
ax.set_xlabel('Iter Number')
ax.set_ylabel('Ek')
plt.show()

print("迭代次数：", it)
print("学习率：", eta)
