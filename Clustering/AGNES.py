import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.io import loadmat
from sklearn.decomposition import PCA

def distEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2))) # 求两个向量之间的距离
# 最小距离
def dist_min(Ci, Cj):
    return np.min([distEclud(i, j) for i in Ci for j in Cj])


# 最大距离
def dist_max(Ci, Cj):
    return np.max([distEclud(i, j) for i in Ci for j in Cj])


# 平均距离
def dist_avg(Ci, Cj):
    return np.sum([distEclud(i, j) for i in Ci for j in Cj]) / (len(Ci) * len(Cj))


# 找到距离最小的下标
def find_Min(M):
    min = 100000
    x = 0
    y = 0
    for i in range(len(M)):
        for j in range(len(M[i])):
            if i != j and M[i][j] < min:
                min = M[i][j]
                x = i
                y = j
    return (x, y, min)


# 簇数转标签
def C2labels(data, C):
    label = np.empty([1, data.shape[0]])
    for i in range(len(C)):
        for j in range(len(C[i])):
            for k in range(data.shape[0]):
                if all(C[i][j] == data[k]):
                    label[0][k] = i
    return label


# 算法模型：
def diy_hierarchical_clustering(data, method,cn):
    length = len(method)
    label = np.empty([length, data.shape[0]])
    for o in range(length):
        # 初始化 C 和 M
        dist = eval(method[o])
        C = []
        M = []
        for i in data:
            Ci = []
            Ci.append(i)
            C.append(Ci)
        for i in C:
            Mi = []
            for j in C:
                Mi.append(dist(i, j))
            M.append(Mi)

        q = len(data)
        # 合并更新
        while q > cn:
            x, y, min = find_Min(M)
            C[x].extend(C[y])
            del (C[y])
            M = []
            for i in C:
                Mi = []
                for j in C:
                    Mi.append(dist(i, j))
                M.append(Mi)
            q -= 1
        label[o] = C2labels(data, C)
    return label, method
def normalize_f(data):
    return (data - data.mean())/data.std()
# data = pd.read_csv("xigua.txt", header=None, names=['A', 'B'])
# data = pd.read_csv("wine.data", header=None)
data_train = pd.read_csv('../other/diabetes_train.txt', sep=' ', header=None)
data_s = data_train.iloc[:, 1:data_train.shape[1]]
# data_s = normalize_f(data_s)
n = 2
x = np.asarray(data_s.values)

pca = PCA(n_components=2)
newX = pca.fit_transform(x)
# x = loadmat("ex7data2.mat")
# x = x['X']
g, me = diy_hierarchical_clustering(newX, ['dist_max'], n)
print(g)
c = np.zeros((n, x.shape[0]))
c = c - 1


for i in range(g.shape[1]):
    for k in range(n):
        if g[0, i] == k:
            j = 0
            while c[k, j] != -1:
                j += 1
            c[k, j] = i



ans_x = np.zeros((n, x.shape[0]))
ans = np.zeros(n)
ans_y = np.zeros((n, x.shape[0]))

for i in range(n):
    j = 0

    while c[i, j] != -1:
        ans_x[i, j] = newX[int(c[i, j]), 0]
        ans_y[i, j] = newX[int(c[i, j]), 1]
        j += 1
    if j != x.shape[0]:
        ans[i] = j

fig, ax = plt.subplots(figsize=(12, 8))

ax.scatter(ans_x[0, 0:int(ans[0])], ans_y[0, 0:int(ans[0])], c='r', marker='*')
ax.scatter(ans_x[1, 0:int(ans[1])], ans_y[1, 0:int(ans[1])], c='b', marker='D')
# ax.scatter(ans_x[2, 0:int(ans[2])], ans_y[2, 0:int(ans[2])], c='c', marker='o')
# ax.scatter(ans_x[3, 0:int(ans[3])], ans_y[3, 0:int(ans[3])], c='y', marker='o')
ax.set_xlabel("x1")
ax.set_ylabel("y1")
ax.set_title("AGNES - max")
plt.show()
