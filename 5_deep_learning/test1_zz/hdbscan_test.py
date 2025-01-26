#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2024/11/12 16:03
# Module    : hdbscan_test.py
# explain   :

import hdbscan
import numpy as np

# 示例数据：假设数据为二维数组
data = np.array([[1.0, 2.0], [1.5, 1.5], [2.0, 3.0], [3.5, 4.0], [5.0, 5.0]])

# 使用 HDBSCAN 聚类
clusterer = hdbscan.HDBSCAN(min_samples=2, min_cluster_size=2)
clusterer.fit(data)

# 打印结果
print(clusterer.labels_)  # 输出聚类标签
print(clusterer.probabilities_)  # 输出每个数据点属于每个簇的概率
