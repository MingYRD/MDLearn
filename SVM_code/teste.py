import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from ML import ReadData
from matplotlib.font_manager import FontProperties

class SVM:
    def __init__(self, dim, kennel=None, Name='svm'):
        self.dim = dim
        self.w = np.zeros([1, dim])
        self.b = np.zeros([1, 1])
        self.kennel = kennel
        self.LamAndY = np.zeros([1, 1])
        self.data = np.zeros([1, 1])
        self.Name = Name

    def train(self, x, y, epochs=20, callback=None, Lam=None, C_val=np.inf, Log=True):
        N = len(x)
        Mat_dot = np.zeros([N, N])
        Mat_dot_y = np.zeros([N, N])
        for i in range(N):
            for j in range(N):
                if self.kennel is None:
                    Mat_dot[i, j] = np.dot(x[i], x[j])
                else:
                    Mat_dot[i, j] = self.kennel(x[i], x[j])
                Mat_dot_y[i, j] = Mat_dot[i, j] * y[i] * y[j]
        if Lam is None:
            Lam = np.random.rand(N)
            delta = np.dot(Lam, y)
            for i in range(N):
                if delta * y[i] < 0:
                    Lam[i] = Lam[i] - y[i] * delta
                    delta = 0
                    if Lam[i] > C_val:
                        delta = (Lam[i] - C_val) * y[i]
                        Lam[i] = C_val
        if Log:
            print('初始化完毕')
        for epoch in range(epochs):
            for a in range(N):
                for b in range(N):
                    subb = Mat_dot[a, a] - Mat_dot[b, a] * 2 + Mat_dot[b, b]
                    if subb == 0:
                        continue

                    C = - Lam[a] * y[a] - Lam[b] * y[b]
                    Sum_Lam_a = np.dot(Lam, Mat_dot_y[:, a]) - Lam[a] * Mat_dot_y[a, a] - Lam[b] * Mat_dot_y[b, a]
                    Sum_Lam_b = np.dot(Lam, Mat_dot_y[:, b]) - Lam[a] * Mat_dot_y[a, b] - Lam[b] * Mat_dot_y[b, b]

                    suba = y[a] * C * (Mat_dot[a, b] - Mat_dot[b, b]) + 1 - y[a] * y[b] - Sum_Lam_a + y[a] * \
                           y[b] * Sum_Lam_b

                    U = suba / subb
                    V = -(C + U * y[a]) * y[b]
                    rate = 1
                    if U < 0:
                        rate = min(rate, (0 - Lam[a]) / (U - Lam[a]))
                    if V < 0:
                        rate = min(rate, (0 - Lam[b]) / (V - Lam[b]))
                    if U > C_val:
                        rate = min(rate, (C_val - Lam[a]) / (U - Lam[a]))
                    if V > C_val:
                        rate = min(rate, (C_val - Lam[b]) / (V - Lam[b]))
                    Lam[a] = Lam[a] + (U - Lam[a]) * rate
                    Lam[b] = Lam[b] + (V - Lam[b]) * rate
            if callback is not None:
                callback(x, y, Lam, self.dim, self.kennel)
        if self.kennel is None:
            w = np.zeros([1, self.dim])
            for i in range(N):
                w = w + Lam[i] * y[i] * x[i]
            h = []
            for i in range(N):
                if Lam[i] > 1e-9:
                    h.append(np.dot(w, x[i]))
            h = np.array(h)
            self.w = w
            self.b[0][0] = (np.max(h) + np.min(h)) / 2
        else:
            self.LamAndY = []
            for i in range(N):
                self.LamAndY.append(Lam[i] * y[i])
            self.data = x
            h = []
            for i in range(N):
                if Lam[i] < 1e-9:
                    continue
                res = 0
                for lay, vec in zip(self.LamAndY, self.data):
                    res = res - lay * self.kennel(vec, self.data[i])
                h.append(res)
            h = np.array(h)
            self.b = (np.max(h) + np.min(h)) / 2
        if Log:
            print('训练结束')
        return Lam

    def forward(self, x):
        if self.kennel is None:
            return self.w @ x.T + self.b
        else:
            ret = np.zeros([1, len(x)])
            for lay, vec in zip(self.LamAndY, self.data):
                ret = ret + lay * self.kennel(vec, x)
            return ret + self.b

    def save(self):
        np.save('save/' + self.Name + '_LamAndY', np.array(self.LamAndY))
        np.save('save/' + self.Name + '_w', np.array(self.w))
        np.save('save/' + self.Name + '_b', np.array(self.b))

    def load(self, data=None):
        try:
            self.LamAndY = np.load('save/' + self.Name + '_LamAndY.npy')
            self.w = np.load('save/' + self.Name + '_w.npy')
            self.b = np.load('save/' + self.Name + '_b.npy')
        finally:
            self.data = data

    def __call__(self, *args, **kwargs):
        return self.forward(*args)


class TestSVM:
    def __init__(self, N=50):
        x = np.random.rand(N, 2)
        x[:, 1] = x[:, 1] * 2
        y = ((x[:, 0] * 3 - 1) > x[:, 1]) * 2 - 1
        for i in range(len(x)):
            if y[i] < 0:
                x[i][1] = x[i][1] - 0.5
        # y = (np.random.rand(N) > 0.5) * 2 - 1
        self.x = x
        self.y = y
        self.point_x = []
        self.point_y = []
        for a, b in zip(x, y):
            if b == 1:
                self.point_x.append(a.tolist())
            else:
                self.point_y.append(a.tolist())
        self.point_x, self.point_y = map(np.array, (self.point_x, self.point_y))
        self.svm = SVM(2, kennel=self.kennel, Name='small_svm')

    def kennel(self, x, y):
        if len(x.shape) == 1 and len(y.shape) == 1:
            return np.dot(x, y)
        return np.dot(x.reshape([1, -1]), y.T)

    def test(self, train=True):
        plt.ion()
        if train:
            self.svm.train(self.x, self.y, 30, self.callback, C_val=1000)
        conlose = (self.svm(self.x) > 0) * 2 - 1
        print(conlose, self.y)
        print(np.sum(conlose == self.y), '/', len(self.x))
        plt.ioff()

    def callback(self, x, y, Lam, dim, kennel):
        w = np.zeros([1, dim])
        for i in range(len(x)):
            w = w + Lam[i] * y[i] * x[i]

        h = []
        for i in range(len(x)):
            if Lam[i] < 1e-9:
                continue
            res = 0

            for la_, y_, vec in zip(Lam, y, x):
                res = res - la_ * y_ * kennel(vec, x[i])
            h.append(res)
        h = np.array(h)
        b = (np.max(h) + np.min(h)) / 2
        lx = np.arange(0, 1, 0.01)
        ly = -(w[0][0] * lx + b) / w[0][1]
        plt.cla()
        plt.plot(self.point_x[:, 0], self.point_x[:, 1], 'o')
        plt.plot(self.point_y[:, 0], self.point_y[:, 1], 'o')
        plt.plot(lx, ly)
        plt.pause(0.05)
        cal = np.sum(Lam)

        for i in range(len(x)):
            for j in range(len(x)):
                cal = cal - 0.5 * Lam[i] * Lam[j] * y[i] * y[j] * np.dot(x[i], x[j])
        print(cal, w, b)


class TestSVM_Classification:
    def __init__(self, TestNum=1000, ):
        data_train = pd.read_csv('../other/diabetes_train.txt', sep=' ', header=None)
        data_test = pd.read_csv('../other/diabetes_test.txt', sep=' ', header=None)

        def normalize_f(data):
            return (data - data.mean()) / data.std()

        cols = data_train.shape[1]
        X_train = data_train.iloc[:, 1:cols]
        y_train = data_train.iloc[:, 0:1]

        colss = data_test.shape[1]
        X_test = data_test.iloc[:, 2:colss]
        y_test = data_test.iloc[:, 0:1]
        # X_test = normalize_f(X_test)
        self.train_x, self.train_y = X_train, y_train
        self.test_x, self.test_y = X_test, y_test
        self.train_y = self.train_y * 2 - 1
        self.test_y = self.test_y * 2 - 1
        # if TestNum < len(self.test_x):
        #     self.train_x = self.train_x[:TestNum, ]
        #     self.train_y = self.train_y[:TestNum, ]
        self.svm = SVM(len(self.train_x[0]),kennel=self.linearKennel, Name='svm_Classification')
        self.gamma = 1
        self.Lam = np.zeros([1, len(self.train_x)])

    def train(self, epochs=20, C=1e9):  # 训练模型
        self.Lam = self.svm.train(self.train_x, self.train_y, epochs=epochs, callback=self.callback, C_val=C)
        print(np.sum((self.svm(self.train_x) > 0) * 2 - 1 == self.train_y.T), len(self.train_x))

    def trainAgain(self, epochs=20, C=1e9, log=False):  # 在原理基础上继续训练
        self.svm.train(self.train_x, self.train_y, epochs=epochs, callback=self.callback, Lam=self.Lam, C_val=C,
                       Log=log)
        right = np.sum((self.svm(self.train_x) > 0) * 2 - 1 == self.train_y.T)
        print(right, right / len(self.train_x))

    def test(self):  # 用测试集测试模型
        self.svm.data = self.train_x
        hhy = self.svm(self.test_x)
        haty = (hhy > 0) * 2 - 1
        print(haty)
        print(self.test_y.T)
        TP = np.sum((haty == 1) * (self.test_y.T == 1))
        TN = np.sum((haty == -1) * (self.test_y.T == -1))
        FN = np.sum((haty == -1) * (self.test_y.T == 1))
        FP = np.sum((haty == 1) * (self.test_y.T == -1))
        print((TP + TN), (TP + TN) / len(self.test_x))
        print('TP\tFN\t[', TP, ' ', FN, ']')
        print('FP\tTN\t[', FP, ' ', TN, ']')
        print('P =', TP / (TP + FP), ' R=', TP / (TP + FN))
        print("F1=",2*TP/(TP-TN+len(self.svm.data)))
        # Ttot = np.sum(self.test_y == 1)
        # Ftot = np.sum(self.test_y == -1)
        # point = np.array([[p[0], y[0]] for p, y in zip(hhy.T, self.test_y)])
        # l = len(point)
        # for i in range(l):
        #     j = np.argmax(point[i:, 0]) + i
        #     t = point[i].copy()
        #     point[i] = point[j]
        #     point[j] = t
        # cnt_T = 0
        # TPR = []
        # FPR = []
        # for i in range(l):
        #     cnt_T = cnt_T + (point[i][1] == 1) * 1
        #     TPR.append(cnt_T / Ttot)
        #     FPR.append((i - cnt_T + 1) / Ftot)
        # TPR = np.array(TPR)
        # FPR = np.array(FPR)
        # plt.plot(FPR, TPR)
        # font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)
        # plt.title('SVM的ROC曲线', fontproperties=font)
        # plt.xlabel("假正例率", fontproperties=font)
        # plt.ylabel("真正例率", fontproperties=font)
        # print('AUC is :', np.sum((TPR[1:] + TPR[:-1]) * 0.5 * (FPR[1:] - FPR[:-1])))
        # plt.show()
        # return FPR, TPR

    def save(self):  # 保存模型
        np.save('save/' + self.svm.Name + '_Lam.npy', self.Lam)
        self.svm.save()

    def load(self):  # 重载模型
        self.svm.load()
        self.Lam = np.load('save/' + self.svm.Name + '_Lam.npy')

    def kennel(self, x, y):  # 高斯核函数
        if len(x.shape) == 1 and len(y.shape) == 1:
            return np.exp(-np.dot(x - y, x - y) / (2 * self.gamma ** 2))
        return np.exp(- np.sum((x - y) ** 2, axis=1) / (2 * self.gamma ** 2))

    def linearKennel(self, x, y):
        if len(x.shape) == 1 and len(y.shape) == 1:
            return np.dot(x, y)
        return np.dot(x.reshape([1, -1]), y.T)

    def callback(self, x, y, Lam, dim, kennel):  # 回调函数
        cal = np.sum(Lam)
        for i in range(len(x)):
            for j in range(len(x)):
                cal = cal - 0.5 * Lam[i] * Lam[j] * y[i] * y[j] * np.dot(x[i], x[j])
        print(cal)

#测试的小例子
#test = TestSVM(30)
#test.test()

csvm = TestSVM_Classification()
csvm.train(5, 2100)
csvm.test()
