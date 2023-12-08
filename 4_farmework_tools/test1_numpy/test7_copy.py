#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/7 21:11
# Module    : test7_copy.py
# explain   : numpu赋值操作


import numpy as np

A = np.arange(12).reshape((3,4))
print(A)

# 矩阵浅拷贝
B = A
print(B)
print(B is A)

A[0,0] = 666
print(A)
print(B)

# 矩阵深拷贝
C = A.copy()
print(C)
print(C is A)


A[0,1] = 888
print(A)
print(B)
print(C)