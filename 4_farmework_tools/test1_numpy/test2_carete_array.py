#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/7 19:27
# Module    : test2_carete_array.py
# explain   : 使用numpy创建各种各样的矩阵

import numpy as np

print(''' 1. 使用array创建 ''')
print(np.array([[1,2,3], [4,5,6]]))
# 指定矩阵的数据格式
print(np.array([[1,2,3], [4,5,6]], dtype=np.int_).dtype)
print(np.array([[1,2,3], [4,5,6]], dtype=np.float_).dtype)


print('''\n 2.定义多维矩阵 ''')
# 多少少个[]就是几维
print(np.array([1,2,3]).ndim) #1
print(np.array([[1,2,3], [4,5,6]]).ndim)#2
print(np.array([[[1,2,3],[4,5,6],[7,8,9]]]).ndim)  #3

# 定义一个3*4的矩阵
matrix_3_4 = np.zeros((3,4))
print(matrix_3_4)
matrix_3_4_f = np.zeros((3,4), dtype=np.float_)
print(matrix_3_4_f)
matrix_3_4_one = np.ones((3,4))
print(matrix_3_4_one)
print(np.arange(5, 20))
print(np.arange(5, 20, 2))

# 生成数字之后重新定义矩阵的行和列
print(np.arange(12).reshape((3,4)))

# eye 创建对角矩阵数组
var = np.eye(4)
print(var)
var1 = np.array([[1.555, 0, 0., 0.],
                 [0., 1., 0., 0.],
                 [0., 0., 1., 0.],
                 [0., 0., 0., 1.]])
print(var1)





