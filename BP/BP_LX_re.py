import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from BP_General import BP_G

def normalize_f(data):
    return (data - data.min())/(data.max() - data.min())

data_train_a = pd.read_csv('../other/sinc_train.txt', sep=' ', header=None, names=['B', 'A'])
data_test_a = pd.read_csv('../other/sinc_test.txt', sep=' ', header=None, names=['B', 'A'])
# data_train_a = normalize_f(data_train_a)
# data_test_a = normalize_f(data_test_a)
cols = data_train_a.shape[1]
X_train = data_train_a.iloc[:, 1:cols]
y_train = data_train_a.iloc[:, 0:1]
y_train = normalize_f(y_train)
X_train = np.asarray(X_train.values)
y_train = np.asarray(y_train.values)
#
# # test
cols = data_test_a.shape[1]
X_test = data_test_a.iloc[:, 1:cols]
y_test = data_test_a.iloc[:, 0:1]
y_test = normalize_f(y_test)
X_test = np.asarray(X_test.values)
y_test = np.asarray(y_test.values)

it = 2000
every_num = []
every_num.append(1)
every_num.append(15)
every_num.append(8)
every_num.append(1)
bpc = BP_G(eta=0.0015, max_iter=it,  hide_every_num=every_num, hide_num=2)
bpc.init_params(X_train)
bpc.bp_train(X_train, X_test, y_train)
yk = bpc.get_y(X_test)


ek, ek_t = bpc.get_ek_s_t()
# x1 = np.linspace(data_test_a.A.min(), data_test_a.A.max(), 100)
# x1 = normalize_f(x1)
fig, ax = plt.subplots(figsize=(12, 8))  # 分解为两个元组对象
ax.plot(data_test_a.A, yk, 'r', label='p')  # xy，颜色，标签
# ax.plot(data_test.A, yy, 'r', label='p')
ax.scatter(data_test_a.A, y_test, label='Test Data')   # 散点图

ax.legend(loc=2)   # 控制图例的位置为第二象限
ax.set_xlabel('A')
ax.set_ylabel('B')
ax.set_title('BP - Linear')
plt.show()


x = np.linspace(0, it, num=it)
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x, ek, 'b')
ax.plot(x, ek_t, 'g')
ax.set_xlabel('Iter Number')
ax.set_ylabel('Ek')
plt.show()

