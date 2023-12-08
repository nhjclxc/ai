#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/7 20:30
# Module    : test5_merge_opt.py
# explain   : 矩阵的合并操作

import numpy as np

A = np.array([[1,2,3],
              [4,5,6]])
B = np.array([[4,5,6],
              [7,8,9]])

# 将矩阵上下(垂直)合并，元组(A, B)前面的矩阵在上，后面的矩阵在下
# (A, B)表示要合并的所有矩阵
print(np.vstack((A, B)))  # v表示垂直vertical
# 将矩阵元组进行左右(水平)合并
print(np.hstack((A, B)))  #

