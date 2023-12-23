#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/23 22:06
# Module    : test01_KarateClub_GAT.py
# explain   :
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.datasets import KarateClub
from torch_geometric.nn import GATConv

# 加载数据集
dataset = KarateClub()
data = dataset[0]

# 构建模型
class GAT(nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.conv1 = GATConv(dataset.num_features, 8, heads=8, dropout=0.6)

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
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

# 在测试集上评估模型
model.eval()
pred = model(data.x, data.edge_index)
correct = int((pred.argmax(dim=1) == data.y).sum())
acc = correct / data.num_nodes
print(f'Accuracy: {acc:.4f}')
