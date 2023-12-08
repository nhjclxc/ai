#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/7 19:16
# Module    : test1_attr.py
# explain   : numpy矩阵的属性

import numpy as np

# 定义一个4*3的二维数组
arr = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]

# 将二维数组转化为矩阵matrix
matrix = np.array(arr)

print(matrix)
print(type(matrix))
print(matrix.size) # 多少个元素
print(matrix.ndim) # 维度2，只有两个方向上的数据
print(matrix.shape) # (4, 3)  (行, 列)
print(matrix.shape[0]) # 获取行
print(matrix.shape[1]) # 获取列


print(np.array([1,2,3]).ndim) #1
print(np.array([[[1,2,3],[4,5,6],[7,8,9]]]).ndim)  #3





