#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/7 19:44
# Module    : test3_opt.py
# explain   : 使用numpy进行基本的运算

import numpy as np


m1 = np.array([[1,2,3],[4,5,6]])
m2 = np.array([[4,5,6],[7,8,9]])
print(m1)
print(m2)

print(m2 - m1)
print(m2 + m1)
print(m2 * m1)  #  对应位置的元素相乘，并不是矩阵的乘法

print('\n\n')
m11 = np.array([[1,2,3],
                [4,5,6]])  # m*p   2*3
m22 = np.array([[4,5],
                [6,7],
                [8,9]])    # p*n   3*2
print(np.dot(m11, m22)) # m11 * m22  ===>>> 2*2  # np.dot表示将两个参数视为矩阵乘法
print(m11.dot(m22)) #  m11 * m22  ===>>> 2*2  # np.dot表示将两个参数视为矩阵乘法

# 求某个矩阵的元素之和，最大值，最小值
print(np.sum(m22))
print(np.max(m22))
print(np.min(m22))

'''
在 NumPy 中，axis 参数表示沿着数组的哪个轴进行操作。
对于二维矩阵来说，axis 参数通常可以取值为 0 或 1，分别表示沿着行或列的方向进行操作。
'''
print('''\n求某个矩阵不同维度的元素之和，最大值，最小值''')
# axis=0表示沿着列方向进行加法操作， axis=1表示沿着行方向进行加法操作
print(np.sum(m22, axis=0)) # 18， 21
print(np.sum(m22, axis=1)) # 9 13 17
print(np.max(m22, axis=0))
print(np.max(m22, axis=1))
print(np.min(m22, axis=0))
print(np.min(m22, axis=1))



print('\n\n')
A = np.array([[1,2,3,4],
              [5,6,7,8],
              [9,10,11,12]])

# 最小值索引，这个时候是看成一维列表了
print(np.argmin(A))
print(np.argmax(A))

# 求矩阵每个元素的平均值
print(np.mean(A))
print(A.mean())
print(np.average(A))


print('\n\n 取矩阵的转置')
print(A)
print(np.transpose(A))
print(A.T)  #矩阵.T：得到的结果表示矩阵的转置
print(A.dot(A.T))

print('\n\n 取矩阵的逆')
print(A)
# 以下语句输出numpy.linalg.LinAlgError: Singular matrix表示这个矩阵是不可逆的，是一个奇异矩阵
# print(np.linalg.inv(A))

# 创建一个非奇异矩阵
singular_matrix = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])

# 尝试求逆矩阵
try:
    inverse_singular_matrix = np.linalg.inv(singular_matrix)
    print("逆矩阵:\n", inverse_singular_matrix)
except np.linalg.LinAlgError as e:
    print("LinAlgError:", e)







