#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/30 15:18
# Module    : test3.py
# explain   :
import torch
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool

# 假设有一个图数据，节点特征维度为 3，共有 4 个节点，边信息如下
x = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0],
                  [7.0, 8.0, 9.0],
                  [10.0, 11.0, 12.0]])  # 节点特征

edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]])  # 边信息

data = Data(x=x, edge_index=edge_index)
# 对图中节点特征进行全局平均池化
output = global_mean_pool(data.x, batch=data.batch)

print(output)
