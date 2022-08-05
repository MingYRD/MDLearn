import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class BP:
    def __init__(self, q=3, l=1, a=None, b=None, v=None, w=None, theta=None, r=None, yk=None, yk_m=None, ek=None,
                 eta=0.1, max_iter=10, ek_s=None):
        """

        :param q: 隐藏层层数
        :param l: 输出层数
        :param a: 隐藏层输入
        :param b: 输出层输入
        :param v: 输入-隐藏权重
        :param w: 隐藏-输出权重
        :param theta: 隐藏阈值
        :param r: 输出阈值
        :param yk: 输出值
        :param yk_m: 真实值
        :param ek: 误差
        :param eta: 学习率
        :param max_iter:迭代次数
        :param ek_s=None: 记录每次迭代的误差
        """
        self.q = q
        self.l = l
        self.a = a
        self.b = b
        self.v = v
        self.w = w
        self.r = r
        self.theta = theta
        self.ek = ek
        self.yk = yk
        self.yk_m = yk_m
        self.eta = eta
        self.max_iter = max_iter
        self.ek_s = ek_s
        self.ek_t = None

    def init_params(self, x):
        n, m = x.shape

        self.v = np.zeros((m, self.q))
        self.w = np.zeros((self.q, self.l))
        self.a = np.zeros(self.q)
        self.b = np.zeros(self.l)
        self.r = np.zeros(self.q)
        self.theta = np.zeros(self.l)
        self.yk = np.zeros((n, self.l))
        self.yk_m = np.zeros((n, self.l))
        self.ek = np.zeros(n)
        self.ek_s = np.zeros(self.max_iter)
        self.ek_t = np.zeros(self.max_iter)

        for i in range(m):
            for j in range(self.q):
                self.v[i, j] = random.random()
        for i in range(self.q):
            for j in range(self.l):
                self.w[i, j] = random.random()

        for i in range(self.q):
            self.a[i] = random.random()
            self.r[i] = random.random()
        for i in range(self.l):
            self.b[i] = random.random()
            self.theta[i] = random.random()

    def _compute_yk_m(self, y):
        for i in range(y.shape[0]):
            # if y[i] == 0:
            #     self.yk_m[i, 0] = 1
            #     self.yk_m[i, 1] = 0
            # else:
            #     self.yk_m[i, 0] = 0
            #     self.yk_m[i, 1] = 1
            self.yk_m[i, 0] = y[i]

    def forward(self, x,  k):
        d = len(x)

        for h in range(self.q):
            pre_num = 0.
            for i in range(d):
                pre_num += self.v[i, h]*x[i]
            self.a[h] = pre_num

        for j in range(self.l):
            pre_num = 0.
            for h in range(self.q):
                pre_num += self.w[h, j]*self.sigmoid(self.a[h] - self.r[h])
            self.b[j] = pre_num

        # for k in range(n):
        pre_num = 0
        for j in range(self.l):
            self.yk[k, j] = self.sigmoid(self.b[j] - self.theta[j])
            pre_num += np.power((self.yk[k, j] - self.yk_m[k, j]), 2)
        self.ek[k] = pre_num / 2

    def backward(self, x, k):
        g = np.zeros(self.l)
        e = np.zeros(self.q)
        d = len(x)

        for j in range(self.l):
            g[j] = self.yk[k, j]*(1 - self.yk[k, j])*(self.yk_m[k, j] - self.yk[k, j])

        for h in range(self.q):
            pre_sum = 0.
            for j in range(self.l):
                pre_sum += self.w[h, j] * g[j]
            e[h] = self.sigmoid(self.a[h] - self.r[h])*(1 - self.sigmoid(self.a[h] - self.r[h]))*pre_sum

        for h in range(self.q):
            # self.r[h] += -self.eta*e[h]
            for j in range(self.l):
                self.w[h, j] += self.eta*g[j]*self.sigmoid(self.a[h] - self.r[h])
        for i in range(d):
            for h in range(self.q):
                self.v[i, h] += self.eta*e[h]*x[i]
        for j in range(self.l):
            self.theta[j] += -self.eta*g[j]
        for h in range(self.q):
            self.r[h] += -self.eta*e[h]

    def get_ek_s(self):
        return self.ek_s, self.ek_t

    def sigmoid(self, z):
        return 1 / (np.exp(-z) + 1)

    def d_sigmoid(self, z):
        return self.sigmoid(z)*(1 - self.sigmoid(z))

    def bp_train(self, x, x_test, y):
        self._compute_yk_m(y)
        n = x.shape[0]
        # pre_ek = 9999

        for iter in range(self.max_iter):
            print(iter)
            for k in range(n):
                self.forward(x[k], k)
                self.backward(x[k], k)
            # if np.sum(self.ek) / n < pre_ek:
            #     pre_ek = np.sum(self.ek) / n
            # else:
            #     break
            self.ek_s[iter] = np.sum(self.ek) / n
            for t in range(x_test.shape[0]):
                self.forward(x_test[t], t)
            self.ek_t[iter] = np.sum(self.ek[0:x_test.shape[0]]) / x_test.shape[0]

    def get_y(self, x_test):
        n, m = x_test.shape
        ykk = []
        for i in range(n):
            self.forward(x_test[i], i)
            ykk.append(self.yk[i, 0])
        return ykk

    # def predict(self, x_test, y_test):
    #     n, m = x_test.shape
    #     ans = 0
    #     y_pred = []
    #     for i in range(n):
    #         self.forward(x_test[i], n)
    #         if self.yk[i, 0] >= 0.5:
    #             y_pred.append(1)
    #             if 1 == y_test[i]:
    #                 ans += 1
    #         else:
    #             y_pred.append(0)
    #             if 0 == y_test[i]:
    #                 ans += 1
    #     return ans / n, y_pred


#
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


maxy = np.max(y_train) + 0.2
miny = np.min(y_train) - 0.2

y_train = y_train / (maxy - miny) - miny / (maxy - miny)
y_test = y_test / (maxy - miny) - miny / (maxy - miny)


it = 1000
bpc = BP(q=15, max_iter=it, eta=0.0015)
bpc.init_params(X_train)
bpc.bp_train(X_train, X_test, y_train)
yk = bpc.get_y(X_test)


ek, ek_t = bpc.get_ek_s()
x1 = np.linspace(data_test_a.A.min(), data_test_a.A.max(), y_test.shape[0])
# x1 = normalize_f(x1)
fig, ax = plt.subplots(figsize=(12, 8))  # 分解为两个元组对象
ax.plot(x1, yk, 'r', label='p')  # xy，颜色，标签
ax.scatter(X_test, y_test, label='Test Data')   # 散点图

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

