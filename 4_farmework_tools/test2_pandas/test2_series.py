#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/8 10:18
# Module    : test2_series.py
# explain   : Series序列相关
import pandas as pd
import numpy as np

'''
Pandas Series 类似表格中的一个列（column），类似于一维数组，可以保存任何数据类型。
'''

'''
创建 Series： 可以使用 pd.Series() 构造函数创建一个 Series 对象，传递一个数据数组（可以是列表、NumPy 数组等）和一个可选的索引数组。
    pandas.Series( data, index, dtype, name, copy)
        data：一组数据(ndarray 类型)。
        index：数据索引标签，如果不指定，默认从 0 开始。
        dtype：数据类型，默认会自己判断。
        name：设置名称。
        copy：拷贝数据，默认为 False。
'''
lst = [1, 2, True, None, np.NaN, np.nan, 1.23]
s1 = pd.Series(lst, name='序列111')
print(s1)

# 测试拷贝属性
s2 = pd.Series(lst, name='序列111', copy=False)  # 表示深拷贝，自己创建数据
print(s2)
lst[2] = '你好，世界'
print(lst)
print(s2)
s3 = pd.Series(lst, name='序列222', copy=True)  # 表示浅拷贝，使用了索引
print(s3)
lst[2] = '哈哈哈哈哈'
print(lst)
print(s3)

# 可以指定序列的索引值

a = ["Google", "Runoob", "Wiki"]
myvar = pd.Series(a, index = ["x1", "y2", "z3"])
print(myvar)
# 通过索引值读取数据
print(myvar['x1'])
# print(myvar['www']) # KeyError: 'www'


# 使用 key/value 对象，类似字典来创建 Series
d = {1: 'xxx', 2: 'zzz', 3: 'aaa', 4: 'qqq', 5 : 'www'}
# 字典key作为索引，value作为值
print(pd.Series(d))
# 只去除对应索引的值
print(pd.Series(d, index=[1,3]))

for item in pd.Series(d).values:
    print(f'item = {item}')



# 创建一个日期格式的序列
s1 = pd.date_range('20231208', '20231215')
print(s1)
s2 = pd.date_range('20231208', periods=6)
print(s2)
print(s2[:2])

print(s2.values)
print(s2.ndim)
print(s2.shape)  # (6,1)因为在pd里面数据是一列下来的，所以是6列



# Series对象基本运算
lst = [0, 1,2,3,4,5,6,7,8,9]
s3 = pd.Series(lst)
# 所有元素*2
print(s3 * 2)
# 对每个元素进行开方操作
print(np.sqrt(s3))
# 序列元素大于5的保留
print(s3[ s3 > 5])


print('''\n\n基本属性''')
s3 = pd.Series(lst, name='基本属性测试')
i = s3.index
print(i)
print(i.start)
print(i.stop)
print(i.step)

print(s3.values)
print(s3.describe())
print(s3.idxmax())  # 最大值索引
print(s3.idxmin())







