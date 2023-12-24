#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/24 10:27
# Module    : test02_gat_reg2.py
# explain   : 使用GAT实现节点推荐


import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

'''
节点推荐是一个重要的图数据任务，它涉及预测图中某个节点与其他节点的关系。以下是一个简单示例，展示如何使用 GAT（Graph Attention Network）进行节点推荐。
首先，我们创建一个简单的图结构数据，然后利用 GAT 模型来预测节点之间的关系。

这个示例创建了一个简单的图结构数据，并使用 GAT 模型对节点之间的关系进行预测。在实际的应用中，可以根据预测结果进行节点推荐或者边的预测。
'''
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

# # 创建图结构数据
# edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
# x = torch.randn(3, 16)  # 3个节点，每个节点特征维度为16
#
# # 指定边缘的正负样本标签（用于节点推荐）
# edge_label = torch.tensor([1, 0, 1, 0], dtype=torch.float)
#
# data = Data(x=x, edge_index=edge_index, edge_label=edge_label)
#
# # 构建 GAT 模型
# class GAT(nn.Module):
#     def __init__(self):
#         super(GAT, self).__init__()
#         self.conv1 = GATConv(data.num_features, 8, heads=2)  # 多头注意力机制
#
#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         return x
#
# model = GAT()
#
# # 定义损失函数和优化器
# criterion = nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01)
#
# # 训练模型
# model.train()
# optimizer.zero_grad()
# out = model(data.x, data.edge_index)
# loss = criterion(out, data.edge_label.view(-1, 1))
# loss.backward()
# optimizer.step()
#
# # 在测试集上评估模型
# model.eval()
# pred = torch.sigmoid(model(data.x, data.edge_index))
# print(f'Predictions: {pred}')
#
# # 可以根据预测结果进行节点推荐或边预测
# # 定义训练循环
# def train(model, criterion, optimizer, data, epochs):
#     model.train()
#     for epoch in range(epochs):
#         optimizer.zero_grad()
#         out = model(data.x, data.edge_index)
#         loss = criterion(out, data.edge_label.view(-1, 1))
#         loss.backward()
#         optimizer.step()
#         print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')
#
# # 训练模型
# model = GAT()
# criterion = nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01)
# train(model, criterion, optimizer, data, epochs=100)





from torch_geometric.datasets import KarateClub

# 加载数据集
dataset = KarateClub()
data = dataset[0]


# 创建一个简单的 GAT 模型
class GATModel(nn.Module):
    def __init__(self):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(dataset.num_features, 16, heads=8, dropout=0.6)
        self.conv2 = GATConv(16 * 8, dataset.num_classes, heads=1, concat=True, dropout=0.6)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = nn.functional.elu(x)
        x = nn.functional.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# 创建模型实例
model = GATModel()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# 模型训练
def train(model, optimizer, criterion, data, epochs):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        # 注意：这里假设数据集的标签位于 data.y 中
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')


# 训练模型
train(model, optimizer, criterion, data, epochs=100)


# 在测试集上评估模型
# 在模型训练后使用模型进行节点预测
def predict_node_class(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)  # 预测节点的类别
        return pred


# 调用预测函数
predictions = predict_node_class(model, data)
print(predictions)

print(data.y)

print(data.y == predictions)

corr = (data.y == predictions).sum()

print(corr/len(data.y))
