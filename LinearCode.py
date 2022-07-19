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


"""
Parameter
"""
# def parameterSolve(X, y):


"""
Data pr
"""

data = pd.read_csv('ex1data1.txt', header=None, names=['Population', 'Profit'])
# print(data.head())
# print(data.describe())


data.insert(0, 'Ones', 1)
print(data.head())
cols = data.shape[1]
X = data.iloc[:, 0:cols - 1]
y = data.iloc[:, cols - 1:cols]

X = np.matrix(X.values)
y = np.matrix(y.values)

theta = np.matrix(np.array([0, 0]))

alpha = 0.01
iters = 10000

g, costs = GradientDescent(X, y, theta, alpha, iters)

print(g)

x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = g[0, 0] + g[0, 1]*x

fig, ax = plt.subplots(figsize=(12, 8))  # 分解为两个元组对象
# ax.plot(x, f, 'r', label='p')  # xy，颜色，标签
ax.scatter(data.Population, data.Profit, label='Traning Data')   # 散点图
ax.legend(loc=2)   # 控制图例的位置为第二象限
ax.set_xlabel('A')
ax.set_ylabel('P')
ax.set_title('Linear')
plt.show()

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(np.arange(iters), costs, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()