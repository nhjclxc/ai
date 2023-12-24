#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/24 10:49
# Module    : test03_gat_rec.py
# explain   :


import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

# 创建一个简单的图数据（这是一个随意构造的示例）
edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
                            [1, 0, 2, 1, 3, 2]], dtype=torch.long)
x = torch.randn(4, 16)  # 4个节点，每个节点的特征向量长度为16

# 创建一个简单的节点推荐任务
# 假设节点0和1是我们关心的节点，我们希望模型推荐与它们最相似的节点
labels = torch.tensor([1, 1, 0, 0], dtype=torch.long)  # 假设节点0和1是我们关心的节点

data = Data(x=x, edge_index=edge_index, y=labels)

# 创建一个简单的 GAT 模型
class GATModel(nn.Module):
    def __init__(self):
        super(GATModel, self).__init__()
        print('data.num_features = ', data.num_features)
        # data.num_features = 16也就是x = torch.randn(4, 16)里面的16
        self.conv1 = GATConv(data.num_features, 8, heads=2)
        self.conv2 = GATConv(8 * 2, 4, heads=2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = nn.functional.elu(x)
        # nn.functional.dropout 是一个用于执行 dropout 操作的函数。在神经网络中，dropout 是一种常用的正则化技术，它有助于减少过拟合现象，提高模型的泛化能力。
        # 在训练阶段，dropout 会以概率 p 随机将输入张量 x 中的部分元素设为0，这些元素将被随机丢弃。而在测试或推理阶段，self.training 会被设置为 False，dropout 操作不会生效，所有元素都会被保留。
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
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')

# 训练模型
train(model, optimizer, criterion, data, epochs=100)

# 对所有节点进行推荐
def recommend_nodes(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        return out.argmax(dim=1)

# 调用推荐函数
recommendations = recommend_nodes(model, data)
print("Recommendations:", recommendations)


