#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/24 10:24
# Module    : test01_gat_reg.py
# explain   : gat实现节点回归预测



import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

'''
当创建一个图结构数据进行节点预测时，你可以模拟一个简单的社交网络，并使用 GAT（Graph Attention Network）进行节点分类。
以下是一个示例，展示如何使用 PyTorch Geometric 创建一个简单的图结构数据，然后使用 GAT 对节点进行预测：

在这个示例中，我们手动创建了一个简单的图结构数据，并使用了一个简单的 GAT 模型对其进行训练和预测。
实际中，你可以使用更复杂的图结构和更大规模的数据集来训练和测试 GAT 模型。
'''
# 创建一个简单的图结构数据（社交网络）
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
x = torch.randn(3, 16)  # 3个节点，每个节点特征维度为16
y = torch.tensor([0, 1, 0], dtype=torch.long)  # 3个节点的标签（两个类别）

data = Data(x=x, edge_index=edge_index, y=y)

# 构建 GAT 模型
class GAT(nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.conv1 = GATConv(data.num_features, 8, heads=2)  # 多头注意力机制

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return x

model = GAT()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
model.train()
for i in range(50):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out, data.y)
    loss.backward()
    optimizer.step()

# 在测试集上评估模型
model.eval()
pred = model(data.x, data.edge_index)
correct = int((pred.argmax(dim=1) == data.y).sum())
acc = correct / data.num_nodes
print(f'Accuracy: {acc:.4f}')
