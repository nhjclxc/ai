#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/22 19:05
# Module    : test01_KarateClub.py
# explain   : 论文分类

# https://www.bilibili.com/video/BV18V411u7nR/?p=14

import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv

# https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html?highlight=Planetoid
# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/planetoid.html#Planetoid
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

# 1. 获取数据
dataset = Planetoid('../../data/Planetoid', name='Cora', transform=NormalizeFeatures())
print(f'dataset = {dataset.data}')
print(f'num of node = {dataset.num_node_features}')
print(f'num of features = {dataset.num_features}')
print(f'num of classes = {dataset.num_classes}')

# print(len(dataset)) 1
# 只有一个数据，即只有一个图，索引使用dataset[0]取出这个图
data = dataset[0]

print(f'data {data}')
# data Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708])
# x=[2708, 1433]：2708个节点，每一个节点是一个1433维度的向量
# edge_index=[2, 10556]：边索引，总共有10556条边
# y=[2708]：里面存储着每一个论文的分类标签，论文引用数据集是一个7分类任务，因此y里面存在着0-6这7个元素
# train_mask=[2708]、val_mask=[2708]、test_mask=[2708]：分别标记了训练数据、验证数据、测试数据

print(data.train_mask.sum().item())
print(data.val_mask.sum().item())
print(data.test_mask.sum().item())

print(torch.unique(data.y))

edge_index = data.edge_index
print('edge_index ', edge_index)

# 2. 定义模型
import torch.nn as nn


class GCN(nn.Module):
    def __init__(self, embedding_dim, output_dim):
        super(GCN, self).__init__()
        # 是 PyTorch 中用来设置随机数种子的函数。设置种子能够使得每次运行程序时产生的随机数保持一致，从而方便调试和复现实验结果。
        # # 后续的随机数生成会使用相同的种子，从而保持一致性
        # # 在同样的种子下，即使有随机操作，也会得到相同的结果
        torch.manual_seed(888)

        # 定义2个卷积层
        self.conv1 = GCNConv(embedding_dim, 128)
        self.conv2 = GCNConv(128, 32)

        # 定义输出层
        self.linear = Linear(32, output_dim)

    def forward(self, node_embedding, edge_index):
        # 图卷积神经网络GCN必须要有两个输入，第一个是节点嵌入，第二个是节点的边表示
        y = torch.relu(self.conv1(node_embedding, edge_index))
        y = torch.relu(self.conv2(y, edge_index))

        # 线性层做分类输出
        y = self.linear(y)
        # 由于输入x是一个x=[2708, 1433]的矩阵，2708表示节点个数，1433表示每一个节点的嵌入表示维度
        # 在经过gcn传播之后，2708没有变，但是1433减少到了标签个数7。传播变化过程：1433 -> 512 -> 128 -> 32 -> 7
        # 故此时的y为 y=[2708, 7]
        # dim=0表示节点个数，而dim=1表示的是节点嵌入的向量。故softmax的dim=1
        output = torch.softmax(y, dim=1)  # 输出层激活函数

        return output


model = GCN(data.num_features, dataset.num_classes)

lr = 0.01
loss_fun = nn.CrossEntropyLoss()  # Define loss criterion.
optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Define optimizer.


# 3. 训练
def predict(model, data, data_mask, is_test=True):
    output = model.forward(data.x, data.edge_index)
    pred = torch.argmax(output, dim=1)

    counter = 0
    counter_true = 0
    for i, flag in enumerate(data_mask):
        if flag:
            if pred[i] == data.y[i]:
                counter_true += 1
            counter += 1
    print(f'验证总数 {counter}， 正确数{counter_true}，正确率 {counter_true/counter}')

    # correct = (pred[data_mask] == data.y[data_mask]).sum().item()
    correct = (pred[data_mask] == data.y[data_mask]).sum()
    acc = int(correct) / int(data_mask.sum())
    print(f'2 验证总数 {data_mask.sum()}， 正确数{correct}，正确率 {acc}')



def train(model, loss_fun, optimizer, data, epochs):
    model.train() # 这个方法用于将模型设置为训练模式
    for epoch in range(epochs):
        # 梯度置零，防止梯度累加
        model.zero_grad()
        # x加入神经网络传播
        output = model.forward(data.x, data.edge_index)
        # 计算损失
        loss = loss_fun(output[data.train_mask], data.y[data.train_mask])  # semi-supervised
        # 梯度反向传播
        loss.backward()
        # 根据损失函数的梯度来更新模型的参数
        optimizer.step()

        if epoch % 10 == 0:
            print(f'epochs: {epoch}，Loss: {loss.item() :.4f}')
            predict(model, data, data.train_mask, is_test=False)


train(model, loss_fun, optimizer, data, 100)

model.eval() # 这个方法用于将模型设置为评估模式。
print('\n\n---------------测试-------------------')
predict(model, data, data.test_mask)
print('---------------验证-------------------')
predict(model, data, data.val_mask)

