import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.decomposition import PCA


def K_means_cluster(x, n):
    n_samples, n_features = x.shape

    # 记录分类后的索引
    C = np.zeros((n, n_samples))
    C = C - 1
    # 记录中心点的坐标
    u_end = np.zeros((n, n_features))
    # 随机指定中心
    for i in range(n):
        k = np.random.randint(0, n_samples)
        for j in range(n_features):
            u_end[i, j] = np.double(x[k, j])
        C[i, 0] = k

    while True:
        g_count = 0
        u = np.zeros((n, n_features))
        for i in range(n):
            for j in range(n_features):
                u[i, j] = u_end[i, j]

        for i in range(n_samples):
            d = np.zeros(n)
            for j in range(n):
                for k in range(n_features):
                    d[j] = d[j] + (x[i, k] - u[j, k]) ** 2
                d[j] = np.sqrt(d[j])
            if np.min(d) != 0:
                idx = np.argmin(d)
                a = 0
                while C[idx, a] != -1:
                    a += 1
                C[idx, a] = i

        for i in range(n):
            count = 0
            sum1 = np.zeros(n_features)
            while C[i, count] != -1:
                # print(C[i, count])
                for j in range(n_features):
                    sum1[j] = sum1[j] + x[int(C[i, count]), j]
                count += 1
            if count != 0:
                for j in range(n_features):
                    u[i, j] = sum1[j] / count


        op = 0
        for i in range(n):
            for j in range(n_features):
                if u[i, j] == u_end[i, j]:
                    op += 1
                else:
                    op = op
        if op == n * n_features:
            break
        else:
            for i in range(n):
                for j in range(n_features):
                    u_end[i, j] = u[i, j]

        C = np.zeros((n, n_samples))
        C = C - 1
    return u_end, C

def normalize_f(data):
    return (data - data.mean())/data.std()
# # data = pd.read_csv("xigua.txt", header=None, names=['A', 'B'])
data_train = pd.read_csv('../diabetes_train.txt', sep=' ', header=None)
# # data = pd.read_csv("wine.data", header=None)
data_s = data_train.iloc[:, 1:data_train.shape[1]]
data_y = data_train.iloc[:, 0:1]
# # data_s = normalize_f(data_s)
# data_s = data_train.iloc[:, 1:data_train.shape[1]]
data_s = normalize_f(data_s)
x = np.asarray(data_s.values)
y = np.asarray(data_y)
# print(x.shape)

x = x - np.mean(x, axis=0)
pca = PCA(n_components=2)
newX = pca.fit_transform(x)
# print(newX.shape)
# x = loadmat("ex7data2.mat")
# x = x['X']
n = 2
u, C = K_means_cluster(newX, n)
print(u)
print(C)
y_pre = np.zeros(y.shape[0])
for i in range(n):
    j = 0
    while C[i, j] != -1:
        if i == 0:
            y_pre[int(C[i, j])] = 1
        j = j + 1
prec = 0
for i in range(y.shape[0]):
    if y_pre[i] == y[i]:
        prec += 1
print(prec / y.shape[0])

ans_x = np.zeros((n, x.shape[0]))
ans = np.zeros(n)
ans_y = np.zeros((n, x.shape[0]))

for i in range(n):
    j = 0

    while C[i, j] != -1:
        ans_x[i, j] = newX[int(C[i, j]), 0]
        ans_y[i, j] = newX[int(C[i, j]), 1]
        j += 1
    if j != x.shape[0]:
        ans[i] = j

fig, ax = plt.subplots(figsize=(12, 8))
for i in range(n):
    ax.scatter(u[i, 0], u[i, 1], c='k', marker='+', linewidths = 10)

ax.scatter(ans_x[0, 0:int(ans[0])], ans_y[0, 0:int(ans[0])], c='r', marker='*')
ax.scatter(ans_x[1, 0:int(ans[1])], ans_y[1, 0:int(ans[1])], c='b', marker='D')
# ax.scatter(ans_x[2, 0:int(ans[2])], ans_y[2, 0:int(ans[2])], c='c', marker='o')
# ax.scatter(ans_x[3, 0:int(ans[3])], ans_y[3, 0:int(ans[3])], c='y', marker='o')
ax.set_xlabel("x1")
ax.set_ylabel("y1")
ax.set_title("k-means")
plt.show()

# 指标计算

# avg
# dist = np.zeros(n)
# for i in range(n):
#     for j in range(int(ans[i])):
#         for k in range(j + 1, int(ans[i])):
#             dist[i] += np.sqrt(np.sum((x[int(C[i, j])] - x[int(C[i, k])]) ** 2))
#
#     dist[i] = 2 * dist[i] / (ans[i]) * (ans[i] - 1)
#
# # print(dist)
#
# # decn
# dist_u = np.zeros((n, n))
# for i in range(n):
#     for j in range(n):
#         dist_u[i, j] = np.sqrt((u[i, 0] - u[j, 0])**2 + (u[i, 1] - u[j, 1])**2)
#
# DBI = 0
# for i in range(n):
#     the_max = 0
#     for j in range(n):
#         if i != j:
#             center_dist = dist_u[i, j]
#             tmp_max = (dist[i] + dist[j]) / center_dist
#             if tmp_max > the_max:
#                 the_max = tmp_max
#     DBI += the_max
# DBI = DBI / n
# print(DBI)