"""
kmeans聚类+网格搜索
"""
import re
import math
import warnings
import tqdm
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from sympy import symbols, Eq, solve
from matplotlib.pyplot import MultipleLocator
from sklearn.preprocessing import MinMaxScaler, Binarizer
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

warnings.filterwarnings("ignore")

best_labels_best = []
silhouette_best = 0
ch_index_best = 0
for i in range(0, 100): # 100个不同的初始化条件
    # 示例数据
    data = # 一个numpy/pandas的矩阵，但是0和1构成的矩阵，且有多行多列，每一行代表一个样本点

    # 定义所需的聚类算法和参数
    clustering_algorithms = [("K-Means", KMeans())]  # 不需要指定初始参数

    # 定义KMeans的超参数搜索范围
    param_grid = {
        "n_clusters": [None, 3, 4, 5],  # 尝试不同的聚类数量
        "init": [None, "k-means++", "random"],  # 不同的初始化方法
        "max_iter": [None, 1, 5, 10, 25, 50, 100],  # 不同的最大迭代次数
        "random_state": [None, 0, 1, 2],
    }

    # 创建GridSearchCV对象
    grid_search = GridSearchCV(estimator=KMeans(), param_grid=param_grid, cv=5)

    # 执行网格搜索
    grid_search.fit(data)

    # 输出最佳参数
    print("Best Parameters:", grid_search.best_params_)

    # 获取最佳参数的KMeans模型
    best_kmeans_model = grid_search.best_estimator_

    # 使用最佳模型进行聚类
    best_kmeans_model.fit(data)
    best_labels = best_kmeans_model.labels_

    # 计算评估指标
    silhouette = silhouette_score(data, best_labels)
    ch_index = calinski_harabasz_score(data, best_labels)

    # print("Clustering Algorithm: K-Means (Tuned)")
    # print("Clustering Labels:", best_labels)
    # print("Silhouette coefficient:", silhouette)
    # print("Calinski-Harabasz index:", ch_index)
    if silhouette > silhouette_best:
        silhouette_best = silhouette.copy()
        best_labels_best = best_labels.copy()
        ch_index_best = ch_index.copy()
        print(
            f"Clustering Algorithm: K-Means (Tuned), Clustering Labels: {best_labels}, Silhouette coefficient: {silhouette}, Calinski-Harabasz index: {ch_index}"
        )

print(
    f"Clustering Algorithm: K-Means (Tuned), Clustering Labels: {best_labels_best}, Silhouette coefficient: {silhouette_best}, Calinski-Harabasz index: {ch_index_best}"
)

# 存储最优初始化参数
import json

data = {"best_labels": best_labels_best.tolist()}

# 保存为JSON文件
with open("best_labels.json", "w") as json_file:
    json.dump(data, json_file)
# 要在其他Python脚本中读取这个JSON文件，你可以使用json.load()函数。
