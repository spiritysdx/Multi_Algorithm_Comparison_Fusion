"""
比较 K-Means（K 均值聚类）、Hierarchical Clustering（层次聚类）、Birch（平衡迭代规约和聚）、Spectral Clustering（谱聚类）
早期探索看哪个方法的聚类效果好，后期进行参数优化
"""
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch, SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# 示例数据
data = # 一个numpy/pandas的矩阵，但是0和1构成的矩阵，且有多行多列，每一行代表一个样本点

for c in [3,4,5]: # 聚类中心数量
    print('*'*50,c,'*'*50)
    clustering_algorithms = [
        ("K-Means", KMeans(n_clusters=c)),
        ("Hierarchical Clustering", AgglomerativeClustering(n_clusters=c)),
        ("Birch", Birch(n_clusters=c)),
        ("Spectral Clustering", SpectralClustering(n_clusters=c))
        ]

    # 迭代聚类算法并分别计算评估指标
    for algorithm_name, algorithm in clustering_algorithms:
        algorithm.fit(data)
        labels = algorithm.labels_

        silhouette = silhouette_score(data, labels)
        ch_index = calinski_harabasz_score(data, labels)

        print("Clustering Algorithm:", algorithm_name)
        print("Clustering Labels:", labels)
        print("Silhouette coefficient:", silhouette)
        print("Calinski-Harabasz index:", ch_index)
        print("---------------------------------------")