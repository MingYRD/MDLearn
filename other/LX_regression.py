import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
COST
"""
def ComputeCost(theta, X, y):
    inner = np.power(X@theta.T - y, 2)
    return np.sum(inner)/2*len(X)


"""
GradientDescent
"""
def GradientDescent(X, y, theta, alpha, iters):
    p = theta.shape[1]
    temp = np.matrix(np.zeros(theta.shape))
    cost = np.zeros(iters)
    for i in range(iters):
        error = (X * theta.T) - y

        for j in range(p):
            inn = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - alpha*np.sum(inn)/len(X)
        theta = temp
        cost[i] = ComputeCost(theta, X, y)

    return theta, cost

def normalize_f(data):
    return (data - data.mean())/data.std()

def MSE(y_pred, y):
    return np.sum(np.power((y_pred - y), 2)) / y_pred.shape[0]
"""
data pr
"""

data_train_a = pd.read_csv('sinc_train.txt', sep=' ', header=None, names=['B', 'A'])
data_test_a = pd.read_csv('sinc_test.txt', sep=' ', header=None, names=['B', 'A'])

x_A = data_train_a.iloc[:, 1:2]
for i in range(5):
    data_train_a.insert(i+2, 'data'+str(i), np.power(x_A, (i+1)*2))
print(data_train_a.head())
data_train = normalize_f(data_train_a)
data_train.insert(1, 'ones', 1)
print(data_train.head())

# test
x_A = data_test_a.iloc[:, 1:2]
for i in range(5):
    data_test_a.insert(i+2, 'data'+str(i), np.power(x_A, (i+1)*2))
# print(data_test_a.head())
data_test = normalize_f(data_test_a)
data_test.insert(1, 'ones', 1)
# print(data_train.head())


cols = data_train.shape[1]
X_train = data_train.iloc[:, 1:cols]
y_train = data_train.iloc[:, 0:1]
X_train = np.matrix(X_train.values)
y_train = np.matrix(y_train.values)

# test
cols = data_test.shape[1]
X_test = data_test.iloc[:, 1:cols]
y_test = data_test.iloc[:, 0:1]
X_test = np.matrix(X_test.values)
y_test = np.matrix(y_test.values)


theta = np.matrix(np.array([0, 0, 0, 0, 0, 0, 0]))

#
alpha = 0.1
iters = 200000
g, cost = GradientDescent(X_train, y_train, theta, alpha, iters)
print(g)
# gx = (X_train.T*X_train).I*X_train.T*y_train
# g = gx.T
x1 = np.linspace(data_test_a.A.min(), data_test_a.A.max(), 100)
x2 = normalize_f(np.power(x1, 2))
x3 = normalize_f(np.power(x1, 4))
x4 = normalize_f(np.power(x1, 6))
x5 = normalize_f(np.power(x1, 8))
x6 = normalize_f(np.power(x1, 10))
x1 = normalize_f(x1)
f = g[0, 0] + g[0, 1]*x1 + g[0, 2]*x2 + g[0, 3]*x3 + g[0, 4]*x4 + g[0, 5]*x5 +g[0, 6]*x6
# f = g[0, 0] + g[0, 1]*x1 + g[0, 2]*x2 + g[0, 3]*x3 + g[0, 4]*x4 + g[0, 5]*x5 +g[0, 6]*x6

fig, ax = plt.subplots(figsize=(12, 8))  # 分解为两个元组对象
ax.plot(x1, f, 'r', label='p')  # xy，颜色，标签
ax.scatter(data_test.A, data_test.B, label='Test Data')   # 散点图
# ax.scatter(data_train.A, data_train.B, label='Traning Data')   # 散点图
ax.legend(loc=2)   # 控制图例的位置为第二象限
ax.set_xlabel('B')
ax.set_ylabel('A')
ax.set_title('Linear')
plt.show()

# print(MSE(f, y_test))
# fig, ax = plt.subplots(figsize=(12, 8))  # 分解为两个元组对象
# ax.plot(x1, MSE(f, y_test), 'r', label='p')  # xy，颜色，标签
# ax.legend(loc=2)   # 控制图例的位置为第二象限
# ax.set_xlabel('B')
# ax.set_ylabel('A')
# ax.set_title('MSE')
# plt.show()

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()