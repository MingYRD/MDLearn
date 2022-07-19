import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

    def forward(self, x, k):
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
            self.r[h] += -self.eta*e[h]
            for j in range(self.l):
                self.w[h, j] += self.eta*g[j]*self.sigmoid(self.a[h] - self.r[h])
        for i in range(d):
            for h in range(self.q):
                self.v[i, h] += self.eta*e[h]*x[i]
        for j in range(self.l):
            self.theta[j] += -self.eta*g[j]

    def get_ek_s(self):
        return self.ek_s

    def sigmoid(self, z):
        return 1 / (np.exp(-z) + 1)

    def d_sigmoid(self, z):
        return self.sigmoid(z)*(1 - self.sigmoid(z))

    def bp_train(self, x, y):
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

    def predict(self, x_test, y_test):
        n, m = x_test.shape
        ans = 0
        y_pred = []
        for i in range(n):
            self.forward(x_test[i], n)
            if self.yk[i, 0] >= 0.5:
                y_pred.append(1)
                if 1 == y_test[i]:
                    ans += 1
            else:
                y_pred.append(0)
                if 0 == y_test[i]:
                    ans += 1
        return ans / n, y_pred


data_train = pd.read_csv('../diabetes_train.txt', sep=' ', header=None)
data_test = pd.read_csv('../diabetes_test.txt', sep=' ', header=None)
data_train_l = data_train
data_test_l = data_test
def normalize_f(data):
    return (data - data.mean())/data.std()


cols = data_train.shape[1]
X_train = data_train.iloc[:, 1:cols]
y_train = data_train.iloc[:, 0:1]
# X_train = normalize_f(X_train)
X_train = np.asarray(X_train.values)
y_train = np.asarray(y_train.values)

colss= data_test.shape[1]
X_test = data_test.iloc[:, 2:colss]
y_test = data_test.iloc[:, 0:1]
# X_test = normalize_f(X_test)
X_test = np.asarray(X_test.values)
y_test = np.asarray(y_test.values)


bpc = BP(q=16, max_iter=200)
bpc.init_params(X_train)
bpc.bp_train(X_train, y_train)
ek = bpc.get_ek_s()
ans, y_pred = bpc.predict(X_test, y_test)
print(ans)
print(y_pred)
# x = np.linspace(0, 1000, num=1000)
# print(x)
# fig, ax = plt.subplots(figsize=(12, 8))
# ax.plot(x, ek, 'b')
# ax.set_xlabel('Iter Number')
# ax.set_ylabel('Ek')
# plt.show()

