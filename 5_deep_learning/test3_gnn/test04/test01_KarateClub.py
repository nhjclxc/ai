#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/22 19:05
# Module    : test01_KarateClub.py
# explain   : 空手道阵营分类
import time

# https://www.bilibili.com/video/BV18V411u7nR/?p=12
# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/karate.html#KarateClub

import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv

# 官方 https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/karate.html#KarateClub
from torch_geometric.datasets import KarateClub

# 获取数据
dataset = KarateClub()
print(f'dataset = {dataset.data}')
print(f'num of node = {dataset.num_node_features}')
print(f'num of features = {dataset.num_features}')
print(f'num of classes = {dataset.num_classes}')

data = dataset[0]
print(f'data {data}')
# x=[34, 34]：第一维度34表示34行即有34个样本M，第二个34表示每一个样本都是34维的向量 即每一个样本的特征向量
# edge_index=[2, 156]：两行，第一行是source节点，第二行表示target目标节点。156表示总共有156条边的连接关系
# y=[34]，因为总共任务是一个4分类任务，因此对应的索引表示的就是某个节点，里面存储的值就表示这个节点是属于哪一类的，有0，1，2，3
# train_mask=[34]标记哪些是训练数据，索引表示节点，里面存储着True或False，True表示训练使用，会利用这些节点来调节可学习参数
# Data(x=[34, 34], edge_index=[2, 156], y=[34], train_mask=[34])

edge_index = data.edge_index
print(edge_index)
# print(edge_index.t())



# GCN模型定义
class GCN(torch.nn.Module):
    def __init__(self, features_num):
        super().__init__()
        # torch.manual_seed(1234)
        # 就是每一个节点的维度，在这里是34
        # 4表示4维的
        self.conv1 = GCNConv(features_num, 32)
        # 在经过一个4*4的卷积
        self.conv2 = GCNConv(32, 16)
        # 在经过一个4*4的卷积
        self.conv3 = GCNConv(16, 8)
        # 最后经过一个MLP输出，
        # dataset.num_classes就是分类的类别，这里就是4
        self.classifier = Linear(8, 4)

    def forward(self, x, edge_index):
        # 输入特征与邻接矩阵（注意格式，上面那种
        # y = self.conv1(x, edge_index)
        y = torch.tanh(self.conv1(x, edge_index))
        y = torch.tanh(self.conv2(y, edge_index))
        y = torch.tanh(self.conv3(y, edge_index))
        # 在分类层
        out = self.classifier(y)
        return out


model = GCN(data.num_features)
print(model)

y = model.forward(data.x, data.edge_index)

# visualize_embedding(h, color=data.y)

loss_fun = torch.nn.CrossEntropyLoss()  # Define loss criterion.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Define optimizer.


def predict(model, data, is_train = False):
    out = model.forward(data.x, data.edge_index)
    sf = torch.softmax(out, dim=1)
    pred = torch.argmax(sf, dim=1)
    # print('pred = ', pred)
    # print('y[i] = ', data.y)
    # if not is_train:
    #     print('train_mask[i] = ', data.train_mask)

    counter = 0
    counter_true = 0
    for i, flag in enumerate(data.train_mask):
        if flag:
            # 训练集验证
            if not is_train:
                print(f'{i}, 还是原来的分类吗？ {pred[i] == data.y[i]}', pred[i], data.y[i])
        else:
            # 预测
            if not is_train:
                print(f'{i}, ======== 分类正确了吗？ {pred[i] == data.y[i]}', pred[i], data.y[i])
            if pred[i] == data.y[i]:
                counter_true += 1
            counter += 1

    print(f'验证总数 {counter}， 正确数{counter_true}，正确率 {counter_true/counter}')


def train(model, loss_fun, optimizer, data, epochs):
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model.forward(data.x, data.edge_index)
        loss = loss_fun(out[data.train_mask], data.y[data.train_mask])  # semi-supervised
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch: {epoch}，Loss: {loss.item() :.4f}')
            predict(model, data, is_train=True)

train(model, loss_fun, optimizer, data, 500)



predict(model, data)

# torch.save(model, 'model.pth')
# model = torch.load('model.pth')

def test_softmax():
    sf = torch.softmax(torch.tensor([0.0416, 0.0215, 0.0250, 0.0235]), dim=0)
    print(sf)
    # 定义一个包含 logits 的张量
    logits = torch.tensor([2.0, 1.0, 0.1])
    # 使用 softmax 函数将 logits 转换为概率分布
    probabilities = torch.softmax(logits, dim=0)
    print(probabilities)
    # 获取概率最大的类别索引
    predicted_class = torch.argmax(probabilities)
    print(predicted_class.item())  # 输出预测的类别索引
