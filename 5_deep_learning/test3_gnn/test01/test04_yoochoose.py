#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/23 14:29
# Module    : test04_yoochoose.py
# explain   : 电商购买预测


from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# 读取数据
df = pd.read_csv('../../data/yoochoose/yoochoose-clicks.dat', header=None)
df.columns = ['session_id', 'timestamp', 'item_id', 'category']

buy_df = pd.read_csv('../../data/yoochoose/yoochoose-buys.dat', header=None)
buy_df.columns = ['session_id', 'timestamp', 'item_id', 'price', 'quantity']

item_encoder = LabelEncoder()
df['item_id'] = item_encoder.fit_transform(df.item_id)
print(df.head())
'''
每一个session_id对应一个用户， item_id表示商品id
    session_id     timestamp             item_id    category
0           1  2014-04-07T10:51:09.277Z     2053        0
1           1  2014-04-07T10:54:09.868Z     2052        0
2           1  2014-04-07T10:54:46.998Z     2054        0
3           1  2014-04-07T10:57:00.306Z     9876        0
4           2  2014-04-07T13:56:37.614Z    19448        0

'''

# 由于数据过大，因此采样部分数据进行学习训练测试
sampled_session_id = np.random.choice(df.session_id.unique(), 10000, replace=False)
df = df.loc[df.session_id.isin(sampled_session_id)]
print(df.nunique())
'''
session_id    10000
timestamp     36068
item_id        8171
category         45
dtype: int64
'''

# 拿到所有的标签
df['label'] = df.session_id.isin(sampled_session_id)
print(df.head())

# ·咱们把每一个session id都当作一个图,每一个图具有多个点和一个标签
# ·其中每个图中的点就是其item_id，特征咱们暂且用其id来表示，之后会做embedding

