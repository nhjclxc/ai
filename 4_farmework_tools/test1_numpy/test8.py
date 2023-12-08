#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/7 21:31
# Module    : test8.py
# explain   :  菜鸟教程：https://www.runoob.com/numpy/numpy-tutorial.html


import numpy as np

'''
属性	            说明
ndarray.ndim	秩，即轴的数量或维度的数量
ndarray.shape	数组的维度，对于矩阵，n 行 m 列
ndarray.size	数组元素的总个数，相当于 .shape 中 n*m 的值
ndarray.dtype	ndarray 对象的元素类型
ndarray.itemsize	ndarray 对象中每个元素的大小，以字节为单位
ndarray.flags	ndarray 对象的内存信息
ndarray.real	ndarray元素的实部
ndarray.imag	ndarray 元素的虚部
ndarray.data	包含实际数组元素的缓冲区，由于一般通过数组的索引获取元素，所以通常不需要使用这个属性。
'''


A = np.arange(12).reshape(3,4)
print(A)

# 从已有矩阵中创建一个新的矩阵
print(np.asarray(A))

# 使用流的形式创建矩阵
s = b'hello world'
C = np.frombuffer(s, dtype = 'S1') #必须指名数据类型
print(C)

# 从可迭代的对象上创建一个矩阵
lst = range(12)
it = iter(lst)

D = np.fromiter(it,dtype=np.int_).reshape(3,4)
print(D)
