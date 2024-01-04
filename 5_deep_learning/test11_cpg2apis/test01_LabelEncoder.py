#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/23 14:42
# Module    : test01_LabelEncoder.py
# explain   : 对标签进行编码
import torch
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd


'''
将labels可以类比为接口api，对所有的api接口进行编码，以获得每一个api的索引，即获得了api集合的词汇表Vocab
'''
''' 对标签进行编码 '''



def test1():
    ''' 不会按照出现的次数来排序 '''

    # 假设有一个分类特征
    labels = ['cat', 'dog', 'dog', 'dog', 'mouse', 'cat', 'dog']

    # 创建 LabelEncoder 对象
    label_encoder = LabelEncoder()

    # 对标签进行编码
    encoded_labels = label_encoder.fit_transform(labels)

    # 输出编码后的标签
    print(encoded_labels, type(encoded_labels))  # 输出: [0 1 2 0 1] <class 'numpy.ndarray'>

    # 反转编码得到原始标签
    decoded_labels = label_encoder.inverse_transform(encoded_labels)
    print(decoded_labels)  # 输出: ['cat' 'dog' 'mouse' 'cat' 'dog']

    # 通过 索引 获取 标签
    print(label_encoder.inverse_transform(np.array([1, 2])))

    # 通过 标签 获取 索引
    indices = label_encoder.transform(np.array(['cat', 'mouse']))
    print(indices, type(indices))

    lst = list(indices)
    print(lst, type(lst))

    t = torch.tensor(indices)
    print(t, type(t))


    pass


def test2():
    ''' 按照标签出现的次数，出现次数越多索引越小 '''
    # 假设有一些标签数据
    labels = ['cat', 'dog', 'mouse','mouse','mouse','mouse','mouse', 'cat', 'dog', 'dog']

    # 使用 Pandas 统计每个标签的出现次数，并按照出现次数进行排序
    label_counts = pd.Series(labels).value_counts().sort_values(ascending=False)

    # 重新排列标签，将出现次数多的放在前面
    sorted_labels = label_counts.index.tolist()

    # 创建 LabelEncoder 对象并拟合重新排列后的标签
    label_encoder = LabelEncoder()
    label_encoder.fit_transform(sorted_labels)

    # 获取编码后的标签
    encoded_labels = label_encoder.transform(labels)
    print(encoded_labels)

    pass


if __name__ == '__main__':

    # test1()

    test2()

    pass

