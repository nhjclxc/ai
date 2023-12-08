#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/8 15:14
# Module    : test6_Data_cleaning.py
# explain   :使用pd进行数据清洗

import json

import pandas as pd
import numpy as np

'''
本文使用到的测试数据 property-data.csv ，包含了四种空数据：n/a、NA、—、na
'''

'''
要删除包含空字段的行，可以使用 dropna() 方法
DataFrame.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
参数说明：
    axis：默认为 0，表示逢空值剔除整行，如果设置参数 axis＝1 表示逢空值去掉整列。(0表示某个单元格为空时剔除那一行，1表示剔除那一列)
    how：默认为 'any' 如果一行（或一列）里任何一个数据有出现 NA 就去掉整行，如果设置 how='all' 一行（或列）都是 NA 才去掉这整行。
    thresh：设置需要多少非空值的数据才可以保留下来的。
    subset：设置想要检查的列。如果是多个列，可以使用列名的 list 作为参数。
    inplace：如果设置 True，将计算得到的值直接覆盖之前的值并返回 None，修改的是源数据。
'''
print('\n\n1.Pandas 清洗空值')
df = pd.read_csv('property-data.csv')
print(df)
df2 = df.dropna(axis=0, how='any')  # NaN的那些行将被删除
print(df)
print(df2)

print('\n\n判断那些元素是null')
print(df['NUM_BEDROOMS'])
print(df['NUM_BEDROOMS'].isnull())

# 读取数据时，指定什么样的数据为不符合要求的空数据
# 读取csv文件时，遇到下面列表里面的数据就会将其读取为NaN
na_values = ['NaN', 'na', '--']
df2 = pd.read_csv('property-data.csv', na_values=na_values)
print(df2['NUM_BEDROOMS'])
print(df2['NUM_BEDROOMS'].isnull())

print('\n\n用某个值来填充空值')
print(df2)
df3 = df2.fillna('你好')
print(df3)
# 替换某一列
print(df2['NUM_BEDROOMS'])
df4 = df2['NUM_BEDROOMS'].fillna('哦哦哦')
print(df4)

import pandas as pd

# 第三个日期格式错误
data = {
    "Date": ['2020/12/01', '2020/12/02', '2020/12/26'],
    "duration": [50, 40, 45]
}

df5 = pd.DataFrame(data, index=["day1", "day2", "day3"])

df5['Date'] = pd.to_datetime(df5['Date'])

print(df5.to_string())

df5.loc['day3','duration'] = 666
print(df5.to_string())
# 对df5中duration大于等于50的数据归一化为11
for index in df5.index:
    print(df5.loc[index, 'duration'])
    if df5.loc[index, 'duration'] >= 50:
        df5.loc[index, 'duration'] = 11
    print(df5.loc[index, 'duration'])
print(df5.to_string())


print('\n\n清晰重复数据')

person = {
    "name": ['Google', 'Runoob', 'Runoob', 'Taobao'],
    "age": [50, 40, 40, 23]
}
df6 = pd.DataFrame(person)

print(df6)
# 检测哪些数据重复
print(df6.duplicated())
# 执行删除
df6.drop_duplicates(inplace=True)
print(df6)








