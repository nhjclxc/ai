#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/24 21:05
# Module    : test1.py
# explain   :  节点回归预测

'''
当涉及图节点推荐时，需要准备一个图数据集并进行节点级别的预测。以下是一个示例，展示了如何使用 PyTorch Geometric 库构建一个简单的图神经网络（GAT）模型来进行图节点的推荐。

'''
import torch
from torch_geometric.data import Data
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

# 1. 创建图数据集
edge_index = torch.tensor([[0, 1, 1, 2],
                            [1, 0, 2, 1]], dtype=torch.long)
x = torch.randn(3, 16)  # 3个节点，每个节点的特征向量大小为16
y = torch.tensor([0, 1, 2], dtype=torch.long)  # 节点的标签
num_classes = len(y)

data = Data(x=x, edge_index=edge_index, y=y)


# 2. 模型定义
class GATModel(nn.Module):
    def __init__(self, in_channels, out_channels, heads, num_classes):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(in_channels, out_channels, heads=heads)
        self.fc = nn.Linear(out_channels*2, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


# 3. 模型训练：
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GATModel(in_channels=16, out_channels=8, heads=2, num_classes = num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

model.train()
for epoch in range(100):
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out, data.y)
    loss.backward()
    optimizer.step()


# 4. 模型测试：
model.eval()
with torch.no_grad():
    logits = model(data)
    pred = logits.argmax(dim=1)
    accuracy = (pred == data.y).sum().item() / data.y.size(0)
    print(f'Accuracy: {accuracy * 100:.2f}%')





