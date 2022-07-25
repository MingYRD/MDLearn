import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from BP_General import BP_G

data_train = pd.read_csv('../diabetes_train.txt', sep=' ', header=None)
data_test = pd.read_csv('../diabetes_test.txt', sep=' ', header=None)
data_train_l = data_train
data_test_l = data_test


def normalize_f(data):
    return (data - data.mean()) / data.std()


def normalize_MM(data):
    return (data - data.min()) / (data.max() - data.min())


cols = data_train.shape[1]
X_train = data_train.iloc[:, 1:cols]
y_train = data_train.iloc[:, 0:1]
# X_train = normalize_MM(X_train)
X_train = np.asarray(X_train.values)
y_train = np.asarray(y_train.values)

colss = data_test.shape[1]
X_test = data_test.iloc[:, 2:colss]
y_test = data_test.iloc[:, 0:1]
# X_test = normalize_MM(X_test)
X_test = np.asarray(X_test.values)
y_test = np.asarray(y_test.values)
n = 200

every_num = []
every_num.append(8)
every_num.append(15)
every_num.append(1)
bpc = BP_G(eta=0.01, hide_every_num=every_num, max_iter=n, hide_num=1)
bpc.init_params(X_train)
bpc.bp_train(X_train, X_test, y_train)
ans, y_pred, y_pro = bpc.predict(X_test, y_test)
ans_t, y_pred_t, y_pro_t = bpc.predict(X_train, y_train)
ek, ek_t = bpc.get_ek_s_t()

print(ans)
print(ans_t)
x = np.linspace(0, n, num=n)
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x, ek, 'b')
ax.plot(x, ek_t, 'g')
ax.set_xlabel('Iter Number')
ax.set_ylabel('Ek')
plt.show()
