import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class BP_M3:

    def __init__(self, eta=0.1, s_num=15, max_iter=100):

        self.eta = eta
        self.s_num = s_num

        self.w1 = None
        self.w2 = None
        self.b1 = None
        self.b2 = None
        self.yk_p = None
        self.A = None
        self.Z = None
        self.max_iter = max_iter
        self.ek = np.zeros(self.max_iter)
        self.ek_t = np.zeros(self.max_iter)

    def _sigmoid(self, z):
        return 1 / (np.exp(-z) + 1)

    def grad(self, z):
        return self._sigmoid(z) * (1 - self._sigmoid(z))

    def init_params(self, x):
        n, m = x.shape
        self.w1 = 2 *np.random.random((m, self.s_num))
        self.w2 = 2*np.random.random((self.s_num, 1))
        self.b1 = 0.1*np.ones((self.s_num,))
        self.b2 = 0.1 * np.ones((1,))
        self.A = []
        self.Z = []
        self.yk_p = np.zeros(n)

    def forward(self, x):
        self.A = []
        self.Z = []
        A1 = np.dot(x, self.w1) + self.b1
        self.A.append(A1)
        Z1 = self._sigmoid(A1)
        self.Z.append(Z1)
        A2 = np.dot(Z1, self.w2) + self.b2
        self.A.append(A2)
        Z2 = self._sigmoid(A2)
        self.Z.append(Z2)
        self.yk_p = Z2


    def error_BP(self, x, y):
        S2 = (self.Z[1] - y)*self.grad(self.A[1])
        delta_W2 = np.dot(self.Z[0].T, S2)
        bias2 = S2.sum(axis=0)

        S1 = np.dot(S2, self.w2.T) * self.grad(self.A[0])
        delta_W1 = np.dot(x.T, S1)
        bias1 = S1.sum(axis=0)
        # update
        self.w1 = self.w1 - self.eta * delta_W1
        self.b1 = self.b1 - self.eta * bias1
        self.w2 = self.w2 - self.eta * delta_W2
        self.b2 = self.b2 - self.eta * bias2



    def BP_train(self, x, x_test, y, y_test):
        for iter in range(self.max_iter):
            print(iter)
            self.forward(x)
            self.error_BP(x, y)
            self.ek[iter] = np.sum((y - self.Z[1]) ** 2) / y.shape[0]
            self.forward(x_test)
            self.ek_t[iter] = np.mean((y_test - self.Z[1]) ** 2) / y_test.shape[0]
    def predict(self, x_test):
        self.forward(x_test)
        return self.yk_p

    def get_ek(self):
        return self.ek, self.ek_t

