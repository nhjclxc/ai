#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/7 21:26
# Module    : test1.py
# explain   :

import pandas as pd
import numpy as np

'''
numpy是一个列表形式的矩阵，而pandas则是一个字典形式的矩阵，可以对每一行赋予不同的含义

Pandas 主要引入了两种新的数据结构：DataFrame 和 Series。
Series 是一种类似于一维数组的对象，它由一组数据（各种 Numpy 数据类型）以及一组与之相关的数据标签（即索引）组成。
    Series： 类似于一维数组或列表，是由一组数据以及与之相关的数据标签（索引）构成。Series 可以看作是 DataFrame 中的一列，也可以是单独存在的一维数据结构。
DataFrame 是一个表格型的数据结构，它含有一组有序的列，每列可以是不同的值类型（数值、字符串、布尔型值）。DataFrame 既有行索引也有列索引，它可以被看做由 Series 组成的字典（共同用一个索引）。
    DataFrame： 类似于一个二维表格，它是 Pandas 中最重要的数据结构。DataFrame 可以看作是由多个 Series 按列排列构成的表格，它既有行索引也有列索引，因此可以方便地进行行列选择、过滤、合并等操作。
'''

# 创建一个Series序列
lst = [1,2,True, None, np.NaN, np.nan, 1.23]
s1 = pd.Series(lst)
print(s1)

# 矩阵在pandas里面是叫做 DataFarme 数据框
dict = {
    'num': [1,2,3],
    'float': [1.1,2.2,3.3],
    'string':['qwe','zzz','哈哈哈']
}
# 每一列的个数必须对齐（也就是每一行的列表元素个数要一样），否则出现：ValueError: All arrays must be of the same length
df = pd.DataFrame(dict)
print(df)









