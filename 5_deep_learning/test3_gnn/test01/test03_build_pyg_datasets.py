#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/23 10:19
# Module    : test03_build_pyg_datasets.py
# explain   : 构造pyg数据集


import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder


def test1():
    """
    构造一个图结构的数据集
    """

    # 定义 tensor类型的点
    x = torch.tensor([[2, 1], [5, 6], [3, 7], [12, 0]], dtype=torch.float)
    # 定义 tensor类型的点对应的标签
    y = torch.tensor([0, 1, 0, 1], dtype=torch.float)
    # 定义边
    edge_index = torch.tensor([[0, 1, 2, 0, 3],  # src
                                [1, 0, 1, 3, 2]],  # tag
                                dtype=torch.float)

    # 创建 torch_geometric 的数据类型的图结构
    data = Data(x=x, y=y, edge_index=edge_index)
    # Data(x=[4, 2], edge_index=[2, 5], y=[4])
    # x=[4, 2]点：输入特征4表示点的个数，2表示每一个点的特征维度
    # edge_index=[2, 5]边：2表示的是src和tag，第一行是源节点，第二行是目标节点，存储的是x里面node的索引；5表示这些节点之间的连接边的条数
    # y=[4]标签：节点标签的数量
    print(data)

    pass


def test2():
    ''' 对标签进行编码 '''

    # 假设有一个分类特征
    labels = ['cat', 'dog', 'mouse', 'cat', 'dog']

    # 创建 LabelEncoder 对象
    label_encoder = LabelEncoder()

    # 对标签进行编码
    encoded_labels = label_encoder.fit_transform(labels)

    # 输出编码后的标签
    print(encoded_labels)  # 输出: [0 1 2 0 1]

    # 反转编码得到原始标签
    decoded_labels = label_encoder.inverse_transform(encoded_labels)
    print(decoded_labels)  # 输出: ['cat' 'dog' 'mouse' 'cat' 'dog']

    pass



if __name__ == '__main__':
    # test1()

    test2()

    pass
