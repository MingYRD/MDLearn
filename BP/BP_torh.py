import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
class BPNet(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_out):
        super(BPNet, self).__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_out)

    def forward(self, x):

        x = torch.sigmoid(self.hidden(x))
        out = torch.sigmoid(self.out(x))
        return out
    def predict(self, x):
        pred = self.forward(x)
        ans = []
        for i in pred:
            if i[0]>i[1]:
                ans.append(0)
            else:
                ans.append(1)
        return torch.tensor(ans)
data_train = pd.read_csv('../diabetes_train.txt', sep=' ', header=None)
data_test = pd.read_csv('../diabetes_test.txt', sep=' ', header=None)
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
y_test = np.array(y_test.values)


train_x = np.zeros((X_train.shape[0], X_train.shape[1]))
test_x = np.zeros((X_test.shape[0], X_test.shape[1]))
train_y = np.zeros(y_train.shape[0])
test_y = np.zeros(y_test.shape[0])
for i in range(y_train.shape[0]):
    train_y[i] = y_train[i]
for j in range(y_test.shape[0]):
    test_y[j] = y_test[j]
for i in range(X_train.shape[0]):
    for j in range(X_train.shape[1]):
        train_x[i, j] = X_train[i, j]
for i in range(X_test.shape[0]):
    for j in range(X_test.shape[1]):
        test_x[i, j] = X_test[i, j]


x_train = torch.FloatTensor(train_x)
y_train = torch.LongTensor(train_y)

x_test = torch.FloatTensor(test_x)
y_test = torch.LongTensor(test_y)

n_features = X_train.shape[1]
n_hidden = 16
n_out = 2
lr = 0.01
epochs = 600
bp = BPNet(n_features, n_hidden, n_out)
optimizer = torch.optim.Adam(bp.parameters(), lr=lr)
loss_fun = torch.nn.CrossEntropyLoss()
losses = []
for i in range(epochs):
    y_pred = bp.forward(x_train)
    loss = loss_fun(y_pred, y_train)
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(accuracy_score(bp.predict(x_test), y_test))

