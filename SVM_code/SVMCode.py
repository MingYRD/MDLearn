import copy
import random

import numpy as np
import kernel_func


class SVM_Pratice:

    def __init__(self, C=1.0, gamma=1.0, degree=3, coef=1.0, kernel=None, kkt_tol=1e-3, max_epoch=100):
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef = coef
        self.kernel = kernel
        if self.kernel == "linear":
            self.kernel_func = kernel_func.linear()
        elif self.kernel == "poly":
            self.kernel_func = kernel_func.poly(self.coef, self.degree)
        else:
            self.kernel_func = kernel_func.rbf(self.gamma)

        self.kkt_tol = kkt_tol
        self.max_epoch = max_epoch
        self.w, self.b = None, None
        self.alpha = None
        self.E = None
        self.kk = 1

        self.support_vectors = []
        self.support_vectors_x = []
        self.support_vectors_y = []
        self.support_vectors_alpha = []

    def _gx(self, x):

        if len(self.support_vectors) == 0:
            if x.ndim <= 1:
                return 0
            else:
                return np.zeros(x.shape[0])
        else:
            if x.ndim <= 1:
                wt_x = 0.0  #
            else:
                wt_x = np.zeros(x.shape[0])
            for i in range(len(self.support_vectors)):
                wt_x += self.support_vectors_alpha[i] * self.support_vectors_y[i] * \
                        self.kernel_func(x, self.support_vectors_x[i])
        # if self.kk == 1:
        #     print((wt_x + self.b).shape)
        #     self.kk += 1
        return wt_x + self.b

    def _init_params(self, x_train, y_train):
        n_samples, n_features = x_train.shape
        # self.alpha = np.zeros(n_samples)
        # Lam = np.random.rand(n_samples)
        # delta = np.dot(Lam, y_train)
        # for i in range(n_samples):
        #     if delta * y_train[i] < 0:
        #         Lam[i] = Lam[i] - y_train[i] * delta
        #         delta = 0
        #         if Lam[i] > self.C:
        #             delta = (Lam[i] - self.C) * y_train[i]
        #             Lam[i] = self.C
        self.alpha = np.zeros(n_samples)
        self.w, self.b = np.zeros(n_features), 0.0
        self.E = self._gx(x_train).reshape(n_samples, 1) - y_train

    def _kkt(self, x_1, y_1, alpha_1):
        # k = y_1 * self._gx(x_1)
        # print(k)
        if alpha_1 == self.C:
            return y_1 * self._gx(x_1) <= 1 + self.kkt_tol
        elif alpha_1 == 0:
            return y_1 * self._gx(x_1) >= 1 - self.kkt_tol
        elif alpha_1 > 0 and alpha_1 < self.C:
            return y_1 * self._gx(x_1) >= 1 - self.kkt_tol and y_1 * self._gx(x_1) <= 1 + self.kkt_tol

    def _update(self, x_train, y_train, y):
        y_predict = self._gx(x_train).reshape(x_train.shape[0], 1)
        self.E = y_predict - y

    def _choose_j(self, best_i):
        vaild_j_list = [j for j in range(len(self.alpha)) if self.alpha[j] > 0 and best_i != j]
        if len(vaild_j_list) > 0:
            idx = np.argmax(np.abs(self.E[best_i, 0] - self.E[vaild_j_list, 0]))  # 绝对误差最大
            best_j = vaild_j_list[int(idx)]
        else:
        #
            # idx = list(range(len(self.alpha)))
            # seq = idx[:best_i+1] + idx[:best_i + 1:]
            while True:
                k = np.random.randint(0, len(self.alpha))
                if k != best_i:
                    best_j = k
                    break
            # best_j = best_i + 1
        return best_j
    def _choose(self, i, m):
        j = i  # 排除i
        while (j == i):
            j = int(np.random.uniform(0, m))
        return j
    def examine_example(self, i, y):
        y1 = y[i]
        alpha1 = self.alpha[i]
        E1 = self.E[i]
        r1 = E1 * y1

        if (r1 < -self.kkt_tol and alpha1 < self.C) or (r1 > self.kkt_tol and alpha1 > 0):
            if len(self.alpha[(self.alpha != 0) & (self.alpha != self.C)]) > 1:
                if self.E[i] > 0:
                    i1 = np.argmin(self.E)
                elif self.E[i] <= 0:
                    i1 = np.argmax(self.E)
                # step_result, model = take_step(i1, i, model)
                # if step_result:
                #     return 1, model
                return i1

        #     for i1 in np.roll(np.where((self.alpha != 0) & (self.alpha != self.C))[0],
        #                       np.random.choice(np.arange(m))):
        #         step_result, model = take_step(i1, i2, model)
        #         if step_result:
        #             return 1, model
        #
        #     for i1 in np.roll(np.arange(m), np.random.choice(np.arange(m))):  # 随机选择起始点
        #         step_result, model = take_step(i1, i2, model)
        #         if step_result:
        #             return 1, model
        #
        # return 0, model
    def _clip(self, y_1, y_2, alpha_2_un, alpha_1_old, alpha_2_old):

        if y_1 == y_2:
            L = max(0, alpha_1_old + alpha_2_old - self.C)
            H = min(self.C, alpha_1_old + alpha_2_old)
        else:
            L = max(0, alpha_2_old - alpha_1_old)
            H = min(self.C, self.C + alpha_2_old - alpha_1_old)
        # print(alpha_2_un)
        if alpha_2_un < L:
            alpha_2_new = L
        elif alpha_2_un > H:
            alpha_2_new = H
        else:
            alpha_2_new = alpha_2_un
        # alpha_j_new = [L if alpha_j_un < L else H if alpha_j_un > H else alpha_j_un]
        return alpha_2_new


    def fit(self, x_train, y_train):
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)

        y = y_train*2 - 1
        # y = copy.deepcopy(y_train)
        # y[y == 0] = -1  # 0 -- 1

        self._init_params(x_train, y)

        for epoch in range(self.max_epoch):
            print(epoch)
            for i in range(x_train.shape[0]):
                alpha_1_old, E_1 = self.alpha[i], self.E[i]
                # print(E_1)
                x_1, y_1 = x_train[i], y[i]
                if not self._kkt(x_1, y_1, alpha_1_old):
                    j = self._choose(i, x_train.shape[0])
                    alpha_2_old, E_2 = self.alpha[j], self.E[j]
                    x_2, y_2 = x_train[j], y[j]

                    k_11 = self.kernel_func(x_1, x_1)
                    k_22 = self.kernel_func(x_2, x_2)
                    k_12 = self.kernel_func(x_1, x_2)
                    eta = k_11 + k_22 - 2 * k_12
                    # print(E_1 - E_2)
                    if eta < 1e-3:
                        continue
                    alpha_2_new_u = alpha_2_old + y_2 * (E_1 - E_2) / eta  # 注意y2
                    alpha_2_new = self._clip(y_1, y_2, alpha_2_new_u, alpha_1_old, alpha_2_old)
                    alpha_1_new = alpha_1_old + y_1 * y_2 * (alpha_2_old - alpha_2_new)

                    self.alpha[i], self.alpha[j] = alpha_1_new, alpha_2_new  # 将新结果进行存储
                    alpha_1_d = alpha_1_new - alpha_1_old
                    alpha_2_d = alpha_2_new - alpha_2_old
                    self.w = self.w + alpha_1_d * y_1 * x_1 + alpha_2_d * y_2 * x_2

                    b_1_new = -E_1 - y_1 * k_11 * alpha_1_d - y_2 * k_12 * alpha_2_d + self.b
                    b_2_new = -E_2 - y_1 * k_12 * alpha_1_d - y_2 * k_22 * alpha_2_d + self.b
                    # b_1_new = -self.E[i] - y_1 * k_11 * alpha_1_d - y_2 * k_12 * alpha_2_d + self.b
                    # b_2_new = -self.E[j] - y_1 * k_12 * alpha_1_d - y_2 * k_22 * alpha_2_d + self.b

                    if alpha_1_new > 0 and alpha_1_new < self.C and alpha_2_new > 0 and alpha_2_new < self.C:
                        self.b = b_1_new
                    # elif alpha_2_new > 0 and alpha_2_new < self.C:
                    #     self.b = b_1_new
                    else:
                        self.b = (b_1_new + b_2_new) / 2

                    self.support_vectors = np.where(self.alpha > 1e-3)[0]
                    self.support_vectors_x = x_train[self.support_vectors, :]
                    self.support_vectors_y = y[self.support_vectors]
                    self.support_vectors_alpha = self.alpha[self.support_vectors]

                    self._update(x_train, y_train, y)

    def get_p(self):
        return self.w, self.b

    def pred_prob(self, X_test):
        # X_test = np.asarray(X_test)
        y_test_hat = np.zeros((X_test.shape[0], 2))
        k = self._gx(X_test)
        for i in range(len(k)):
            if k[i] > 0:
                y_test_hat[i, 1] = 1
        #     y_test_hat[i, 1] = self.sigmoid(k[i])
        # y_test_hat[:, 0] = 1.0 - y_test_hat[:, 1]
        # print(y_test_hat)
        return y_test_hat

    def predict(self, X_test):
        y_test_hat = np.zeros((X_test.shape[0], 2))
        # k = self._gx(X_test)

        for i in range(X_test.shape[0]):
            k = 0.
            for j in range(X_test.shape[1]):
                k += X_test[i, j] * self.w[j]
            # print(k)
            k = k + self.b
            if k >= 0:
                y_test_hat[i, 1] = 1
            else:
                y_test_hat[i, 1] = 0
        #     y_test_hat[i, 1] = self.sigmoid(k[i])
        # y_test_hat[:, 0] = 1.0 - y_test_hat[:, 1]
        # print(y_test_hat[:, 1])
        return y_test_hat[:, 1]

    # def predict(self, X_test):
    #     return np.argmax(self.pred_prob(X_test), axis=1)

    def correctRate(self, X_test, y_test):
        pred = self.predict(X_test)
        # print(pred)
        ans = 0
        for i in range(y_test.shape[0]):
            if pred[i] == y_test[i]:
                ans += 1
        return ans / y_test.shape[0]

    @staticmethod
    def sigmoid(z):
        return 1 / (np.exp(-z) + 1)
