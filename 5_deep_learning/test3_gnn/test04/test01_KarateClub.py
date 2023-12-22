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
from torch import nn
import networkx as nx
import matplotlib
# matplotlib.use('Agg')  # 静态画图  指定使用 Agg 后端
import matplotlib.pyplot as plt

plt.switch_backend('Agg')  # 也可以尝试其他的 backend
from torch_geometric.utils import to_networkx

import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv


#
def visualize_graph(G, color):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False, node_color=color, cmap="Set2")
    plt.show()


def visualize_embedding(h, color, epoch=None, loss=None):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])
    h = h.detach().cpu().numpy()
    plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch}，Loss: {loss.item() :.4f}', fontsize=16)
    plt.show()


# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/karate.html#KarateClub

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


# 可视化展示
G = to_networkx(data, to_undirected=True)

# visualize_graph(G, color=data.y)


# GCN模型定义
class GCN(torch.nn.Module):
    def __init__(self, features_num):
        super().__init__()
        # torch.manual_seed(1234)
        # 就是每一个节点的维度，在这里是34
        # 4表示4维的
        self.conv1 = GCNConv(features_num, 8)
        # 在经过一个4*4的卷积
        self.conv2 = GCNConv(8, 8)
        # 在经过一个4*4的卷积
        self.conv3 = GCNConv(8, 8)
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

criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Define optimizer.


for epoch in range(401):
    optimizer.zero_grad()
    out = model.forward(data.x, data.edge_index)  # h是两维向量，主要是为了咱们画个图
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # semi-supervised
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch: {epoch}，Loss: {loss.item() :.4f}')

def predict(model, data):
    out = model.forward(data.x, data.edge_index)
    sf = torch.softmax(out, dim=1)
    pred = torch.argmax(sf, dim=1)
    # print('pred = ', pred)
    # print('y[i] = ', data.y)
    print('train_mask[i] = ', data.train_mask)

    counter = 0
    counter_true = 0
    for i, flag in enumerate(data.train_mask):
        if flag:
            # 训练
            print(f'{i}, 还是原来的分类吗？ {pred[i] == data.y[i]}', pred[i], data.y[i])
        else:
            # 预测
            print(f'{i}, ======== 分类正确了吗？ {pred[i] == data.y[i]}', pred[i], data.y[i])
            if pred[i] == data.y[i]:
                counter_true += 1
            counter += 1

    print(counter_true,counter, counter_true/counter)



predict(model, data)


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
