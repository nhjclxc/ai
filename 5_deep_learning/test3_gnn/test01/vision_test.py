#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/22 21:28
# Module    : vision_test.py
# explain   :

# import numpy as np
# import matplotlib
# matplotlib.use('Agg')  # 或者其他支持的后端 Qt5Agg Agg TkAgg
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
#
# # 假设你有多个 1x4 的张量，存储在 data 中
# # 这里用随机数据作为示例
# data = [np.random.rand(1, 4) for _ in range(10)]
#
# # 将数据转换成 numpy 数组并进行降维
# data_array = np.concatenate(data, axis=0)  # 合并成一个数组
# pca = PCA(n_components=2)  # 降维至二维
# transformed_data = pca.fit_transform(data_array)
#
# # 绘制降维后的数据点
# plt.scatter(transformed_data[:, 0], transformed_data[:, 1])
# plt.title('Visualization of 1x4 Tensors in 2D')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.show()



import matplotlib
matplotlib.use('Agg')  # 或者其他支持的后端 Qt5Agg Agg TkAgg
import matplotlib.pyplot as plt
x = [1,2,3]
y = [5,-7,8]
plt.plot(x, y)
plt.xlabel('x - 轴')
plt.ylabel('y - 轴')
plt.title('简单图形')
plt.show()
import torch

# 假设有两个张量
x = torch.tensor([1, 2, 3, 4])
y = torch.tensor([1, 3, 3, 5])

# 使用 torch.eq() 比较两个张量的元素是否相等
equal_elements = torch.eq(x, y)

# 使用 torch.sum() 计算相等元素的数量
num_equal_elements = torch.sum(equal_elements).item()

print(equal_elements)  # 输出比较结果（True表示对应位置的元素相等，False表示不相等）
print(num_equal_elements)  # 输出相等元素的数量
