import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
class LogisticRe:
    def sigmoid(self, z):
        return 1 / (np.exp(-z)+1)

    def Cost(self,theta, X, y):
        first = np.multiply(-y, np.log(self.sigmoid(X * theta.T)))
        second = np.multiply((1 - y), np.log(1 - self.sigmoid(X * theta.T)))
        return np.sum(first - second) / (len(X))

    def GradientDescent(self,X, y, theta, alpha, iters):
        p = theta.shape[1]
        temp = np.matrix(np.zeros(theta.shape))
        cost = np.zeros(iters)
        for i in range(iters):
            error = self.sigmoid(X * theta.T) - y

            for j in range(p):
                inn = np.multiply(error, X[:, j])
                temp[0, j] = theta[0, j] - alpha*np.sum(inn)/len(X)
            theta = temp
            cost[i] = self.Cost(theta, X, y)

        return theta, cost

    def predict(self,theta, X, y):
        pred = self.sigmoid(X @ theta.T)

        # print(pred.shape[0])
        ans = 0
        for i in range(pred.shape[0]):
            if pred[i, 0] >= 0.5:
                if 1 == y[i, 0]:
                    ans = ans + 1
            else:
                if 0 == y[i, 0]:
                    ans = ans + 1
        return ans / pred.shape[0]

    def PR(self,theta, X, y):

        P = []
        R = []
        pred = self.sigmoid(X @ theta.T)
        k = np.sort(pred.ravel())
        # print(k)
        # print(pred.shape[0])
        for j in range(pred.shape[0]):
            tp = 0
            tn = 0
            fn = 0
            fp = 0
            for i in range(pred.shape[0]):
                if pred[i, 0] >= k[0, j]:
                    if 1 == y[i, 0]:
                        tp = tp + 1
                    else:
                        fp = fp + 1
                else:
                    if 0 == y[i, 0]:
                        tn = tn + 1
                    else:
                        fn = fn + 1

            P.append(tp / (tp + fp))
            R.append(tp / (tp + fn))

        return P, R

    def ROC(self,theta, X, y):
        TPR = []
        FPR = []
        pred = self.sigmoid(X @ theta.T)
        k = np.sort(pred.ravel())[::-1]
        # print(k)
        # print(k.shape[1])
        # print(pred)
        # print(pred.shape[0])
        for j in range(pred.shape[0]):
            tp = 0
            tn = 0
            fn = 0
            fp = 0
            for i in range(pred.shape[0]):
                if pred[i, 0] > k[0, j]:
                # if pred[i, 0] >= 0.5:
                    if y[i, 0] == 1:
                        tp = tp + 1
                    else:
                        fp = fp + 1
                else:
                    if y[i, 0] == 0:
                        tn = tn + 1
                    else:
                        fn = fn + 1
            TPR.append(tp / (tp + fn))
            FPR.append(fp / (tn + fp))

        return TPR, FPR


    def ROCX(self,theta, X, y):
        pred = self.sigmoid(X @ theta.T)
        # k = np.sort(pred.ravel())[::-1]
        # print(k)
        # print(k.shape[1])
        # print(pred)
        # print(pred.shape[0])

        tp = 0
        tn = 0
        fn = 0
        fp = 0
        for i in range(pred.shape[0]):
                # if pred[i, 0] > k[0, j]:
            if pred[i, 0] >= 0.5:
                if y[i, 0] == 1:
                    tp = tp + 1
                else:
                    fp = fp + 1
            else:
                if y[i, 0] == 0:
                    tn = tn + 1
                else:
                    fn = fn + 1


        return tp,tn,fn,fp
    # P, R = PR(g, X_train, y_train)
    def f1_score(self,theta, X, y):
        P = 0.
        R = 0.
        f1 = 0.
        tp = 0
        tn = 0
        fn = 0
        fp = 0
        pred = self.sigmoid(X @ theta.T)
        # print(k)
        # print(pred.shape[0])
        for j in range(pred.shape[0]):
            for i in range(pred.shape[0]):
                if pred[i, 0] >= 0.5:
                    if 1 == y[i, 0]:
                        tp = tp + 1
                    else:
                        fp = fp + 1
                else:
                    if 0 == y[i, 0]:
                        tn = tn + 1
                    else:
                        fn = fn + 1

        P = (tp / (tp + fp))
        R = (tp / (tp + fn))
        f1 = 2 * P * R / (P + R)
        return f1



    def ComputeAUC(self,TPR, FPR):
        auc = 0.
        x = FPR[::-1]
        y = TPR[::-1]
        for i in range(len(FPR)-1):
                auc += (x[i+1] - x[i]) * (y[i] + y[i+1])

        return auc / 2

    def calAUC(self, prob, labels):
        f = list(zip(prob, labels))
        rank = [values2 for values1, values2 in sorted(f, key=lambda x: x[0])]
        rankList = [i + 1 for i in range(len(rank)) if rank[i] == 1]
        posNum = 0
        negNum = 0
        for i in range(len(labels)):
            if (labels[i] == 1):
                posNum += 1
            else:
                negNum += 1
        auc = 0
        auc = (sum(rankList) - (posNum * (posNum + 1)) / 2) / (posNum * negNum)
        print(auc)
        return auc
"""
data pr
"""

# data_train = pd.read_csv('diabetes_train.txt', sep=' ', header=None, names=['Admitted', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
# data_test = pd.read_csv('diabetes_test.txt', sep=' ', header=None, names=['Admitted', 'k','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])

# #
# # positive = data_train[data_train['Admitted'].isin(1)]
# # negative = data_train[data_train['Admitted'].isin(0)]
#
#
# data_train.insert(1, 'Ones', 1)
# cols = data_train.shape[1]
# X_train = data_train.iloc[:, 1:cols]
# y_train = data_train.iloc[:, 0:1]
#
# X_train = np.matrix(X_train.values)
# y_train = np.matrix(y_train.values)
#
#
# data_test.insert(2, 'Ones', 1)
# print(data_test)
# cols = data_test.shape[1]
# X_test = data_test.iloc[:, 2:cols]
# # print(X_test)
# y_test = data_test.iloc[:, 0:1]
# # print(y_test)
# X_test = np.matrix(X_test.values)
# y_test = np.matrix(y_test.values)
#
#
# theta = np.matrix(np.zeros(9))
#
# alpha = 0.03
# iters = 10000
# lg = LogisticRe()
# g, costs = lg.GradientDescent(X_train, y_train, theta, alpha, iters)
# print(g)

# lg = LogisticRegression()
# lg.fit(X_train, y_train)
# print(lg.score(X_test,y_test))
#
# fig, ax = plt.subplots(figsize=(12, 8))
# ax.plot(np.arange(iters), costs, 'r')
# ax.set_xlabel('Iterations')
# ax.set_ylabel('Cost')
# ax.set_title('Error vs. Training Epoch')
# plt.show()



# print(ComputeAUC(TPR, FPR))
#
# fig, ax = plt.subplots(figsize=(12, 8))
# ax.plot(P, R, 'r')
# ax.set_xlabel('P')
# ax.set_ylabel('R')
# ax.set_title('P-R Curve')
# plt.show()
# print(f1_score(g, X_train, y_train))
#
# TPR, FPR = ROC(g, X_train, y_train)
#
# fig, ax = plt.subplots(figsize=(12, 8))
# ax.plot(FPR, TPR, 'r')
# ax.set_xlabel('FPR')
# ax.set_ylabel('TPR')
# ax.set_title('ROC Curve')
# plt.show()



#
# tp,tn,fn,fp = lg.ROCX(g, X_test, y_test)
# print(tp)
# print(tn)
# print(fn)
# print(fp)

# print(predict(g, X_train, y_train))
# print(predict(g, X_test, y_test))
"""
实际验证
"""
# pred = sigmoid(X_train @ g.T)
# fpr, tpr, thersholds = roc_curve(y_train, pred)
# print(auc(fpr, tpr))
# fig, ax = plt.subplots(figsize=(12, 8))
# ax.plot(fpr, tpr, 'r')
# ax.set_xlabel('FPR')
# ax.set_ylabel('TPR')
# ax.set_title('ROC Curve')
# plt.show()

# pred = sigmoid(X_train @ g.T)
# precision, recall, _ = precision_recall_curve(y_train, pred)
# fig, ax = plt.subplots(figsize=(12, 8))
# ax.plot(precision, recall, 'r')
# ax.set_xlabel('P')
# ax.set_ylabel('R')
# ax.set_title('P-R Curve')
# plt.show()

