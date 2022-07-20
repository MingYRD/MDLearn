import random

import numpy as np
import pandas as pd

class BP_G:

    def __init__(self, hide_num=1, hide_every_num=None, out_num=1, eta=0.1, max_iter=10, ek_s=None):
        """

        :param hide_num:隐藏层层数
        :param hide_every_num: 每个隐藏层数量
        :param out_num: 输出层层数
        :param h_in: 每层输入集合
        :param h_out: 每层输出
        :param weight_v: 权重
        :param threshold: 阈值
        :param yk: 真实值
        :param yk_p: 预测值
        :param ek: 误差
        :param eta: 学习率
        :param max_iter: 迭代次数
        :param ek_s: 平均误差
        """
        self.hide_num = hide_num
        if hide_every_num is None:
            self.hide_every_num = []
        else:
            self.hide_every_num = hide_every_num
        self.out_num = out_num
        self.h_in = None
        self.h_out = None
        self.weight_v = None
        self.threshold = None
        self.yk = None
        self.yk_p = None
        self.ek = None
        self.eta = eta
        self.max_iter = max_iter
        self.ek_s = ek_s
        self.ek_t = None

    def init_params(self, x):
        n, d = x.shape

        max_num = np.max(self.hide_every_num)
        self.h_in = np.zeros((self.hide_num+2, max_num))
        self.h_out = np.zeros((self.hide_num+2, max_num))
        self.weight_v = np.zeros((self.hide_num+1, max_num, max_num))
        self.threshold = np.zeros((self.hide_num+2, max_num))
        self.yk = np.zeros((n, self.out_num))
        self.yk_p = np.zeros((n, self.out_num))
        self.ek = np.zeros(n)
        self.ek_s = np.zeros(self.max_iter)
        self.ek_t = np.zeros(self.max_iter)
        for i in range(self.hide_num+2):
            for j in range(max_num):
                self.h_in[i, j] = random.random()
                self.h_out[i, j] = random.random()
                self.threshold[i, j] = random.random()

        for i in range(self.hide_num+1):
            for j in range(max_num):
                for k in range(max_num):
                    self.weight_v[i, j, k] = random.random()

    def _compute_yk(self, y):
        for i in range(y.shape[0]):
            for j in range(self.out_num):
                self.yk[i, j] = y[i, j]

    def _forward(self, x, k):
        for i in range(len(x)):
            self.h_in[0, i] = x[i]
            self.h_out[0, i] = x[i]

        for i in range(self.hide_num+1):
            j = i + 1
            for j_n in range(self.hide_every_num[j]):
                pre_num = 0.
                for i_n in range(self.hide_every_num[i]):
                    pre_num += self.weight_v[i, i_n, j_n]*self.h_out[i, i_n]
                self.h_in[j, j_n] = pre_num
                self.h_out[j, j_n] = self._sigmoid(self.h_in[j, j_n] - self.threshold[j, j_n])
        pre_sum = 0.
        for j in range(self.out_num):
            self.yk_p[k, j] = self.h_out[self.hide_num+1, j]
            # pre_sum += np.power((self.yk_p[k, j] - self.yk[k, j]), 2)
            pre_sum += -(self.yk[k, j] * np.log(self.yk_p[k, j]) + (1 - self.yk[k, j]) * np.log(1 - self.yk_p[k, j]))
        self.ek[k] = pre_sum

    def _backward(self, x, k):
        max_num = np.max(self.hide_every_num)
        f = np.zeros((self.hide_num+1, max_num))
        for i in range(len(x)):
            self.h_in[0, i] = x[i]
            self.h_out[0, i] = x[i]

        for j in range(self.out_num):
            # f[self.hide_num, j] = self.yk_p[k, j]*(1 - self.yk_p[k, j])*(self.yk[k, j] - self.yk_p[k, j])
            f[self.hide_num, j] = self.yk[k, j] - self.yk_p[k, j]

        i = self.hide_num
        while i >= 1:
            for h in range(self.hide_every_num[i]):
                pre_sum = 0.
                for j in range(self.hide_every_num[i+1]):
                    pre_sum += self.weight_v[i, h, j] * f[i, j]
                f[i-1, h] = self.h_out[i, h]*(1-self.h_out[i, h])*pre_sum
            i -= 1

        for i in range(self.hide_num, -1, -1):
            for j in range(self.hide_every_num[i]):
                for l in range(self.hide_every_num[i+1]):
                    self.weight_v[i, j, l] += self.eta*f[i, l]*self.h_out[i, j]

        for i in range(self.hide_num+1, 0, -1):
            for j in range(self.hide_every_num[i]):
                self.threshold[i, j] += -self.eta*f[i-1, j]

    def bp_train(self, x, x_test, y):
        self._compute_yk(y)
        n = x.shape[0]

        for iter in range(self.max_iter):
            print(iter)
            for k in range(n):
                self._forward(x[k], k)
                self._backward(x[k], k)
            self.ek_s[iter] = np.sum(self.ek) / n
            for t in range(x_test.shape[0]):
                self._forward(x_test[t], t)
            self.ek_t[iter] = np.sum(self.ek[0:x_test.shape[0]]) / x_test.shape[0]

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def get_ek_s_t(self):
        return self.ek_s, self.ek_t

    def get_y(self, x_test):
        ykk = []
        for i in range(x_test.shape[0]):
            self._forward(x_test[i], i)
            ykk.append(self.yk_p[i, 0])
        return ykk

    def predict(self, x_test, y_test):
        n, m = x_test.shape
        ans = 0
        y_pred = []
        y_pro = []
        for i in range(n):
            self._forward(x_test[i], i)
            y_pro.append(self.yk_p[i, 0])
            if self.yk_p[i, 0] >= 0.5:
                y_pred.append(1)
                if 1 == y_test[i]:
                    ans += 1
            else:
                y_pred.append(0)
                if 0 == y_test[i]:
                    ans += 1
        return ans / n, y_pred, y_pro
