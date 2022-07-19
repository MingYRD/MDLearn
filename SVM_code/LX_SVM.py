import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from BP.BP_Classification import BP_C
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import sklearn.model_selection as ms
# from LX_logistic import LogisticRe
# from SVMCode import SVM_Pratice
from sklearn import metrics
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report

#
#
# my data:
data_train = pd.read_csv('../diabetes_train.txt', sep=' ', header=None)
data_test = pd.read_csv('../diabetes_test.txt', sep=' ', header=None)
data_train_l = data_train
data_test_l = data_test


def normalize_f(data):
    return (data - data.mean()) / data.std()


def calAUC(prob, labels):
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

    auc = (sum(rankList) - (posNum * (posNum + 1)) / 2) / (posNum * negNum)
    print(auc)
    return auc


def ComputeAUC(TPR, FPR):
    auc = 0.
    x = np.sort(FPR)
    y = np.sort(TPR)
    for i in range(len(FPR) - 1):
        auc += (x[i + 1] - x[i]) * (y[i] + y[i + 1])

    return auc / 2


cols = data_train.shape[1]
X_train = data_train.iloc[:, 1:cols]
y_train = data_train.iloc[:, 0:1]
# X_train = normalize_f(X_train)
X_train = np.asarray(X_train.values)
y_train = np.asarray(y_train.values)

colss = data_test.shape[1]
X_test = data_test.iloc[:, 2:colss]
y_test = data_test.iloc[:, 0:1]
# X_test = normalize_f(X_test)
X_test = np.asarray(X_test.values)
y_test = np.asarray(y_test.values)
# print(X_test.shape)

"""
库函数
"""
#
# svm_old = SVC(C=3000, kernel="linear")
# svm_old.fit(X_train, y_train)
# y_p = svm_old.predict(X_test)
#
# lg = LogisticRegression()
# lg.fit(X_train, y_train)
# #
svm_rbf1 = SVC(probability=True)
# 基于网格搜索获取最优模型

params = [
    {'kernel': ['linear'], 'C': [i for i in np.arange(0.01, 1.01, 0.01)]},
    # {'kernel': ['poly'], 'C': [1, 10], 'degree': [2, 3]},
    {'kernel': ['rbf'], 'C': [i for i in np.arange(0.1, 10, 0.1)],
     'gamma': [i for i in np.arange(0.01, 1.01, 0.01)]}]
svm_rbf = ms.GridSearchCV(estimator=svm_rbf1, param_grid=params, cv=5, scoring='roc_auc')
svm_rbf.fit(X_train, y_train)
# 网格搜索训练后的副产品
print("模型的最优参数：", svm_rbf.best_params_)
print("最优模型分数：", svm_rbf.best_score_)
print("最优模型对象：", svm_rbf.best_estimator_)
print(svm_rbf.score(X_test, y_test))
# 输出网格搜索每组超参数的cv数据
for p, s in zip(svm_rbf.cv_results_['params'], svm_rbf.cv_results_['mean_test_score']):
    print(p, s)


# y_p = svm_rbf.predict(X_test)
# svm_poly.fit(X_train, y_train)
# print(classification_report(y_test, y_p))
# print(svm_rbf.score(X_train, y_train))
# print(svm_rbf.score(X_test, y_test))
# print(AUC(y_test, svm_rbf.decision_function(X_test)))
# FPR_ll, recall_ll, thresholds = roc_curve(y_test, svm_old.decision_function(X_test), pos_label=1)
# print(metrics.auc(FPR_ll,recall_ll))
# print(ComputeAUC(recall_ll, FPR_ll))
# FPR_rbf, recall_rbf, thresholdss = roc_curve(y_test, svm_rbf.decision_function(X_test), pos_label=1)
# print(metrics.auc(FPR_rbf,recall_rbf))
# FPR_lg, recall_lg, thresholdsss = roc_curve(y_test, lg.decision_function(X_test), pos_label=1)
# print(metrics.auc(FPR_lg,recall_lg))
# plt.figure()
# plt.plot(FPR, recall, color='red'
#          ,label='ROC curve (area = %0.2f)' % AUC(y_test, svm_old.decision_function(X_test)))
#
# 逻辑回归

# data_train_l.insert(1, 'Ones', 1)
# cols = data_train_l.shape[1]
# X_train_l = data_train_l.iloc[:, 1:cols]
# y_train_l = data_train_l.iloc[:, 0:1]
#
# X_train_l = np.matrix(X_train_l.values)
# y_train_l = np.matrix(y_train_l.values)
#
# data_test_l.insert(2, 'Ones', 1)
# cols = data_test_l.shape[1]
# X_test_l = data_test_l.iloc[:, 2:cols]
# # print(X_test)
# y_test_l = data_test_l.iloc[:, 0:1]
# # print(y_test)
# X_test_l = np.matrix(X_test_l.values)
# y_test_l = np.matrix(y_test_l.values)
#
#
# theta = np.matrix(np.zeros(9))
# alpha = 0.03
# iters = 10000
# lg = LogisticRe()
# g, costs = lg.GradientDescent(X_train_l, y_train_l, theta, alpha, iters)
# TPR, FPR = lg.ROC(g, X_train_l, y_train_l)
# print(metrics.auc(FPR,TPR))
# bpc = BP_C(q=16, max_iter=200, eta=0.01)
# bpc.init_params(X_train, X_test)
# bpc.bp_train(X_train, y_train, X_test, y_test)
# ans,y_pred, y_pro = bpc.predict(X_test, y_test)
# TPR_bp, FPR_bp = bpc.ROC(X_test, y_test)
# print(metrics.auc(FPR_bp, TPR_bp))
# fig, ax = plt.subplots(figsize=(12, 8))
# ax.plot(FPR_lg, recall_lg, 'r', label='Logistic Regression')
# ax.plot(FPR_ll, recall_ll, 'b', label='SVM - Linear')
# ax.plot(FPR_rbf, recall_rbf, 'g', label='SVM - RBF')
# ax.plot(FPR_bp, TPR_bp, 'k', label='BP - 16')
# ax.legend(loc=2)
# ax.set_xlabel('FPR')
# ax.set_ylabel('TPR')
# ax.set_title('ROC Curve-Compared')
# plt.show()


"""
自编程
"""
#
# svm_new = SVM_Pratice(C=0.03, kernel="line", max_epoch=2000)
# svm_new.fit(X_train, y_train)
# y_test_pred = svm_new.predict(X_train)
#
# print(classification_report(y_test, y_test_pred))
# print(svm_new.correctRate(X_test, y_test))


# y_test_pred = svm_new.predict(X_train)
# print(classification_report(y_train, y_test_pred))
# print(svm_new.predict(X_train))
# # print(y_test)
# print(svm_new.correctRate(X_test, y_test))
