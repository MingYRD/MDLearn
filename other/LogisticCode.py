import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (np.exp(-z)+1)

def Cost(theta, X, y):
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / (len(X))

def GradientDescent(X, y, theta, alpha, iters):
    p = theta.shape[1]
    temp = np.matrix(np.zeros(theta.shape))
    cost = np.zeros(iters)
    for i in range(iters):
        error = sigmoid(X * theta.T) - y

        for j in range(p):
            inn = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - alpha*np.sum(inn)/len(X)
        theta = temp
        cost[i] = Cost(theta, X, y)

    return theta, cost



data = pd.read_csv('ex2data1.txt', header=None, names=['A', 'B', 'C'])
# print(data.head())
# print(data.describe())
positive = data[data['C'].isin([1])]
negative = data[data['C'].isin([0])]


data.insert(0, 'Ones', 1)
print(data.head())
cols = data.shape[1]
X = data.iloc[:, 0:cols - 1]
y = data.iloc[:, cols - 1:cols]

X = np.matrix(X.values)
y = np.matrix(y.values)

theta = np.matrix(np.array([0, 0, 0]))

alpha = 0.1
iters = 100000

g, costs = GradientDescent(X, y, theta, alpha, iters)
print(g)
plotting_x1 = np.linspace(30, 100, 100)
plotting_h1 = (- g[0, 0] - g[0, 1] * plotting_x1) / g[0, 2]

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(plotting_x1, plotting_h1, 'y', label='Prediction')
ax.scatter(positive['A'], positive['B'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['A'], negative['B'], s=50, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
plt.show()
# fig, ax = plt.subplots(figsize=(12, 8))
# ax.plot(np.arange(iters), costs, 'r')
# ax.set_xlabel('Iterations')
# ax.set_ylabel('Cost')
# ax.set_title('Error vs. Training Epoch')
# plt.show()