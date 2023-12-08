#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/8 13:48
# Module    : test4_csv.py
# explain   : pandas操作csv文件数据

import pandas as pd
import numpy as np


# 读取csv文件
csv = pd.read_csv('nba.csv')
# print(csv.to_string())

print('\n\n数据处理')
# head返回前面几行的数据 默认返回前5行数据
print(csv.head())
print(csv.head(3))
# tail返回后面极寒的数据
print(csv.tail())

print('\n\n返回表格的基本信息')
print(csv.info)





# 写csv文件
# dict = {
#     'num': [1, 2, 3],
#     'float': [1.1, 2.2, 3.3],
#     'string': ['qwe', 'zzz', '哈哈哈']
# }
# df = pd.DataFrame(dict)
# df.to_csv('dict.csv')



