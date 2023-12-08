#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/7 20:18
# Module    : test4_index.py
# explain   : numpy创建的矩阵的索引

import numpy as np
A = np.array([[1,2,3,4],
              [5,6,7,8],
              [9,10,11,12]])

print(A)
# 很容易看出numpy生成的矩阵其实是一个多维列表
print(A[0], ' === ')
print(A[1], ' === ')
print(A[2], ' === ')
print(A[2][1], ' === ')
print(A[2,1], ' === ')

print('\n\n 对矩阵的行进行迭代')
for row in A:
    print(row, '--')

print('\n\n 对矩阵的列进行迭代，由于np不支持对矩阵的列进行迭代，'
      '但是我们可以对矩阵先进行转置，然后迭代转置后的矩阵的行就等价于迭代了原始矩阵的列')
for column in A.T:
    print(column, '--')

print('\n\n 对矩阵的每一个值进行迭代')
print(A.flatten())  # 将np的矩阵转化为一个py的列表
print(type(A.flatten()))
for item in A.flat:  # A.flat迭代器
    print(item)


