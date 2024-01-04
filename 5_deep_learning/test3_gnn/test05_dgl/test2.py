#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/29 10:05
# Module    : test2.py
# explain   :
import dgl

import torch
import dgl.nn as dglnn
import torch.nn as nn


# 定义边列表
src = [0, 1, 2, 3]
dst = [1, 2, 3, 0]

# 创建图
g = dgl.graph((src, dst))

input_dim, edge_dim = 16,16
# 为节点添加特征
num_nodes = g.number_of_nodes()
input_feature = torch.randn(num_nodes, input_dim)  # input_dim 是特征的维度
g.ndata['feat'] = input_feature
# 为边添加特征
num_edges = g.number_of_edges()
edge_feature = torch.randn(num_edges, edge_dim)  # edge_dim 是特征的维度
g.edata['feat'] = edge_feature

# 构建图神经网络模型
class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        # 16*20
        self.conv1 = dglnn.GraphConv(input_dim, hidden_dim)
        # 20*5
        self.conv2 = dglnn.GraphConv(hidden_dim, output_dim)

    def forward(self, g, features):
        # 第一层图卷积
        # 4*16
        x = self.conv1(g, features)
        # (4*16) * (16*20) = 4*20
        x = torch.relu(x)
        # 第二层图卷积
        # (4*20) * (20*5) = 4*5
        x = self.conv2(g, x)
        return x


# 使用模型
input_dim = 16
hidden_dim = 20
output_dim = 5

model = GNNModel(input_dim, hidden_dim, output_dim)
node_features = g.ndata['feat']  # 获取节点特征
print(node_features)
output = model(g, node_features)
print(output.shape)
print(output)