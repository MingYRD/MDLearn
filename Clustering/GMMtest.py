import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.io import loadmat


class GMM:

    def __init__(self, u=None, sigma=None, a=None, c=None, ci=None, R=None, k=1, max_iters=100):
        self.u = u
        self.sigma = sigma
        self.a = a
        self.c = c
        self.ci = ci
        self.R = R
        self.k = k
        self.iters = max_iters

    def _init_params(self, x):
        n, m = x.shape
        self.u = np.zeros((self.k, m))
        self.sigma = np.zeros(((self.k, m, m)))
        self.a = np.zeros(self.k)
        self.R = np.zeros((n, self.k))
        self.c = np.zeros((self.k, n))
        self.c = self.c - 1
        self.ci = []

        # self.u[0] = x[5, :]
        # self.u[1] = x[21, :]
        # self.u[2] = x[26, :]
        for i in range(self.k):
            self.a[i] = 1 / self.k
            self.sigma[i] = np.eye(m)*0.1
            rd = random.randint(0, n)
            self.u[i] = x[rd, :]
    def _compute_p(self, x, j, i):
        up = np.exp((-0.5)*(x[j, :] - self.u[i, :])*(np.linalg.inv(self.sigma[i]))*(x[j, :] - self.u[i, :]).T)
        down = np.power(2*np.pi, x.shape[0]/2)*np.sqrt(np.linalg.det(self.sigma[i]))

        return up / down
    def _compute_rji(self,x, j, i):
        up = self.a[i]*self._compute_p(x, j, i)
        down = 0.
        for l in range(self.k):
            down += self.a[l]*self._compute_p(x, j, l)
        # self.R[j, i] = up / down
        return up / down
    def _compute_ui(self, x, i):
        up = np.zeros((1, x.shape[1]))
        down = 0.
        for j in range(x.shape[0]):
            up += x[j, :]*self.R[j, i]
            down += self.R[j, i]
        return up / down
    def _compute_sigmai(self, x, i):
        up = np.zeros((x.shape[1], x.shape[1]))
        down = 0.
        for j in range(x.shape[0]):
            up += self.R[j, i] * (x[j, :] - self.u[i, :]).T*(x[j, :] - self.u[i, :])
            down += self.R[j, i]
        return up / down

    def _compute_ai(self, x, i):
        up = 0.
        down = x.shape[0]
        for j in range(x.shape[0]):
            up += self.R[j, i]
        return up / down

    def fit(self, x):
        self._init_params(x)
        n, m = x.shape
        while self.iters > 0:
            for j in range(n):
                for i in range(self.k):
                    self.R[j, i] = self._compute_rji(x, j, i)

            for i in range(self.k):
                self.u[i] = self._compute_ui(x, i)
            for i in range(self.k):
                self.sigma[i] = self._compute_sigmai(x, i)
            for i in range(self.k):
                self.a[i] = self._compute_ai(x, i)

            self.iters -= 1


        for j in range(n):
            idx = np.argmax(self.R[j, :])
            self.ci.append(idx)
            l = 0
            while self.c[idx, l] != -1:
                l += 1
            self.c[idx, l] = j
    def get_ci(self):
        return self.ci

    def get_c(self):
        return self.c
    def get_u(self):
        return self.u




# data = pd.read_csv("xigua.txt", header=None, names=["a", "b"])
# x = np.matrix(data.values)
x = loadmat("ex7data2.mat")
x = x['X']
x=np.matrix(x)
n = k =3
gmm = GMM(k=3, max_iters=10)
gmm.fit(x)
C = gmm.get_c()
u = gmm.get_u()
print(gmm.get_c())
print(gmm.get_ci())

ans_x = np.zeros((n, x.shape[0]))
ans = np.zeros(n)
ans_y = np.zeros((n, x.shape[0]))

for i in range(n):
    j = 0

    while C[i, j] != -1:
        ans_x[i, j] = x[int(C[i, j]), 0]
        ans_y[i, j] = x[int(C[i, j]), 1]
        j += 1
    if j != x.shape[0]:
        ans[i] = j

fig, ax = plt.subplots(figsize=(12, 8))
for i in range(n):
    ax.scatter(u[i, 0], u[i, 1], c='k', marker='+', linewidths = 10)

ax.scatter(ans_x[0, 0:int(ans[0])], ans_y[0, 0:int(ans[0])], c='r', marker='*')
ax.scatter(ans_x[1, 0:int(ans[1])], ans_y[1, 0:int(ans[1])], c='b', marker='D')
ax.scatter(ans_x[2, 0:int(ans[2])], ans_y[2, 0:int(ans[2])], c='c', marker='o')
# ax.scatter(ans_x[3, 0:int(ans[3])], ans_y[3, 0:int(ans[3])], c='y', marker='o')
ax.set_xlabel("x1")
ax.set_ylabel("y1")
ax.set_title("GMM-100")
plt.show()
