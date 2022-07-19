import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

data_train = pd.read_csv('../diabetes_train.txt', sep=' ', header=None)
data_s = data_train.iloc[:, 1:data_train.shape[1]]
# data_s = normalize_f(data_s)
n = 3
x = np.asarray(data_s.values)

pca = PCA(n_components=2)
newX = pca.fit_transform(x)

clf = AgglomerativeClustering(n_clusters=3, linkage='ward')
clf.fit(newX)



