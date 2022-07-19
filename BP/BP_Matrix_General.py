import numpy as np


class BP_M:

    def __init__(self, hide_num=1, lay=1, eta=0.1, every_num=None, max_iter=100):
        self.hide_num = hide_num
        self.lay = lay
        self.eta = eta
        self.every_num = every_num
        if self.every_num == None:
            self.every_num = []

        self.A = None
        self.Z = None
        self.w = None
        self.b = None
        self.max_iter = max_iter
        self.ek = []
        self.ek_t = []

    def _sigmoid(self, z):
        return 1 / (np.exp(-z) + 1)

    def grad(self, z):
        return self._sigmoid(z) * (1 - self._sigmoid(z))

    def init_params(self, x):
        n, m = x.shape
        # self.A = 2 * np.random.random((m, max_num)) - 1
        self.w = []
        self.b = []
        self.w.append(2 * np.random.random((m, self.every_num[0])) - 1)
        for i in range(len(self.every_num) - 1):
            we = 2 * np.random.random((self.every_num[i], self.every_num[i + 1])) - 1
            self.w.append(we)

        # self.w = 2 * np.random.random((m, max_num)) - 1
        # self.b = 0.1 * np.ones((self.hide_num + 1, max_num))
        for i in range(len(self.every_num)):
            if self.every_num[i] != 1:
                be = 0.1 * np.ones(self.every_num[i])
                self.b.append(be)
            else:
                self.b.append(0.1*np.ones(1))

    def forward(self, x):
        self.A = []
        self.Z = []
        self.A.append(x)
        self.Z.append(x)
        for e in range(self.hide_num + 1):
            Ain = np.dot(self.Z[e], self.w[e]) + self.b[e]
            self.A.append(Ain)
            Zout = self._sigmoid(Ain)
            self.Z.append(Zout)

    def error_BP(self, y):
        delta_w = []
        bias = []
        s = []
        s.append((self.Z[len(self.Z) - 1] - y) * self.grad(self.A[self.hide_num + 1]))
        w_idx = len(self.every_num) - 1
        for e in range(self.hide_num):
            s.append(np.dot(s[e], self.w[w_idx].T) * self.grad(self.A[w_idx]))
            w_idx -= 1
        s_idx = len(s) - 1
        for i in range(self.hide_num+1):
            delta_w.append(np.dot(self.Z[i].T, s[s_idx]))
            bias.append(s[s_idx].sum(axis=0))
            s_idx -= 1
        # j = len(delta_w)-1
        for i in range(self.hide_num + 1):
            self.w[i] -= self.eta * delta_w[i]
            self.b[i] -= self.eta * bias[i]
            # j -= 1

    def BP_train(self, x_train, y_train, x_test, y_test):
        for epoch in range(self.max_iter):
            print(epoch)
            self.forward(x_train)
            self.ek.append(np.mean((self.Z[len(self.Z) - 1] - y_train) ** 2))
            self.error_BP(y_train)

            self.forward(x_test)
            self.ek_t.append(np.mean((self.Z[len(self.Z) - 1] - y_test) ** 2))

    def predict(self, x_test):
        self.forward(x_test)
        yk = self.Z[len(self.Z) - 1]
        return yk

    def get_ek(self):
        return self.ek, self.ek_t
