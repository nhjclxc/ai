#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/7 20:40
# Module    : test6_segment.py
# explain   : 矩阵分割

import numpy as np

A = np.arange(12).reshape((3,4))
print(A)
'''
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
'''
# 分成3行
print(np.split(A, 3, axis=0))
# 分成两列
print(np.split(A, 2, axis=1))




'''
在 NumPy 中，axis 参数表示沿着数组的哪个轴进行操作。
对于二维矩阵来说，axis 参数通常可以取值为 0 或 1，分别表示沿着行或列的方向进行操作。
axis 参数通常可以取值为 0 或 1，分别表示沿着行或列的方向进行操作。
axis 参数通常可以取值为 0 或 1，分别表示沿着行或列的方向进行操作。
axis 参数通常可以取值为 0 或 1，分别表示沿着行或列的方向进行操作。
axis 参数通常可以取值为 0 或 1，分别表示沿着行或列的方向进行操作。
axis 参数通常可以取值为 0 或 1，分别表示沿着行或列的方向进行操作。
'''
# 创建一个二维矩阵
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# 对矩阵进行求和操作，并指定 axis=0 和 axis=1
sum_along_axis_0 = np.sum(matrix, axis=0)  # 沿着行的方向进行求和
sum_along_axis_1 = np.sum(matrix, axis=1)  # 沿着列的方向进行求和

# 输出结果
print("原始矩阵:\n", matrix)
print("\n沿着 axis=0 方向的求和结果（列的和）:\n", sum_along_axis_0) #  [12 15 18]
print("\n沿着 axis=1 方向的求和结果（行的和）:\n", sum_along_axis_1) #  [ 6 15 24]