#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/8 10:49
# Module    : test3_DataFrame.py
# explain   :DataFrame 相关操作

import pandas as pd
import numpy as np

'''
DataFrame 是一个表格型的数据结构，它含有一组有序的列，每列可以是不同的值类型（数值、字符串、布尔型值）。
DataFrame 既有行索引也有列索引，它可以被看做由 Series 组成的字典（共同用一个索引）。

DataFrame 特点：
    列和行： DataFrame 由多个列组成，每一列都有一个名称，可以看作是一个 Series。同时，DataFrame 有一个行索引，用于标识每一行。
    二维结构： DataFrame 是一个二维表格，具有行和列。可以将其视为多个 Series 对象组成的字典。
    列的数据类型： 不同的列可以包含不同的数据类型，例如整数、浮点数、字符串等。


'''

'''
pandas.DataFrame( data, index, columns, dtype, copy)
参数说明：
    data：一组数据(ndarray、series, map, lists, dict 等类型)。
    index：索引值，或者可以称为行标签。
    columns：列标签，默认为 RangeIndex (0, 1, 2, …, n) 。
    dtype：数据类型。
    copy：拷贝数据，默认为 False。
'''

dict = {
    'num': [1, 2, 3],
    'float': [1.1, 2.2, 3.3],
    'string': ['qwe', 'zzz', '哈哈哈']
}
# 每一对key/value表示的是一列，key表示这一列的名字，value里面的列表表示这一列的所有数据，每一列的数据个数必须相同
df = pd.DataFrame(dict)
print(df)
print(pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=['qqq', 'aaa', 'zzz']))

'''
`iloc` 和 `loc` 都是 Pandas 中用于索引和选择数据的方法，但它们有着不同的用途和行为。

### `iloc`：

- `iloc` 是基于 **位置（integer-location）** 进行选择的。它使用整数索引来选择行和列。
- 用法：`iloc[row_index, column_index]`，其中 `row_index` 和 `column_index` 可以是单个整数、整数列表或切片对象。
- `iloc` 接受整数作为参数，例如 `iloc[0, 1]` 表示选择第一行第二列的数据。
- 使用位置索引可以灵活地按照行和列的位置来选择数据。

### `loc`：

- `loc` 是基于 **标签（label）** 进行选择的。它使用行和列的标签来进行选择。
- 用法：`loc[row_label, column_label]`，其中 `row_label` 和 `column_label` 可以是单个标签、标签列表或切片对象。
- `loc` 接受标签作为参数，例如 `loc['A', 'B']` 表示选择行标签为 'A'、列标签为 'B' 的数据。
- 使用标签索引可以根据索引的具体标签名称选择数据，更适用于基于索引标签进行数据选取的情况。

### 区别总结：

- `iloc` 使用整数位置索引，而 `loc` 使用标签索引。
- `iloc` 是基于整数位置进行选择的，`loc` 是基于标签进行选择的。
- `iloc` 适用于按位置选择数据，`loc` 更适用于按标签选择数据。

根据你的需求和数据的索引方式，选择适合的方法来进行数据的选择和操作。
'''
print('\n\n 通过列索引访问数据')
print(df.loc[0])
print(df.loc[[0, 1]])  # 返回多行索引
print(df.loc[0:1])  # 返回多行索引
print('\n\n 通过行索引访问数据')
print(df.iloc[0])

print('\n\n 基本属性和方法')
print(df['string'])
print(df[['float', 'string']])  # 获取多列

print(df.columns)  # 列方向上的索引
print(df.index)  # 行方向上的索引
print(df.shape)
print(df.describe())

print('\n\n数据操作')
print(df)
# 添加新的一列
df['test'] = [111, 222, 333]
print(df)
# 修改某列的值
df['test'] = [666, 888, 999]
print(df)
# df.iloc[行索引, 列索引] = [值]  iloc表示操作索引，值的个数一定要和前面行索引包含的元素个数相同
df.iloc[1: 2, 3] = [555]
print(df)
# 删除一列，axis=1表示删除第0维度，即列方向上的数据，'test'表示列方向上的那一列
# 若要在原始 DataFrame 上直接修改，可以使用 inplace=True 参数，否则就是返回一个删除后的新的df
df.drop('test', axis=1, inplace=True)
print(df)
# 删除一行，返回删除后的df
print(df.drop(1, axis=0))

print('\n排序')
# 排序  ascending=False表示降序， ascending=True表示升序
df.sort_values(by='float', ascending=False, inplace=True)
print(df)

print('\n重命名列')
# {'原列的索引' : '新的索引'}
df.rename(columns={'num': 'int'}, inplace=True)
print(df)


# print('\n\n从外部数据源创建 DataFrame：')
# df_csv = pd.read_csv('CSVFile.csv')
# print(df_csv)
#
# df_xls = pd.read_excel('ExcelFile.xlsx')
# print(df_xls)