#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/24 21:17
# Module    : test2.py
# explain   : 节点推荐



'''
你想要一个推荐模型，它能够基于给定的节点，预测可能连接到这些节点的其他节点。
以下是一个示例代码，使用 PyTorch Geometric 中的 GAT 模型实现节点推荐：
'''

import torch
from torch_geometric.data import Data

# 1. 数据构造

import torch
from torch_geometric.data import Data, Dataset

class CustomDataset(Dataset):
    def __init__(self, num_graphs, num_node, dim_feature, num_edge = None):
        super().__init__()
        self.num_graphs = num_graphs
        self.num_node = num_node
        self.data_list = [self.generate_random_graph(num_node, dim_feature, num_edge) for _ in range(num_graphs)]

    @staticmethod
    def generate_random_graph(num_node, feature_dim = 16, num_edge = None):
        # 随机生成节点特征
        num_nodes = int(torch.randint(1, num_node + 1, ()))
        x = torch.randn(num_nodes, feature_dim)  # 假设节点特征维度为 16
        if num_edge == None:
            num_edge = num_nodes * 2
        edge_index = torch.randint(0, num_nodes, (2, num_edge))  # 随机生成边
        return Data(x=x, edge_index=edge_index)

    def __len__(self):
        return self.num_graphs

    def len(self):
        return self.num_graphs

    def get(self, idx):
        return self.data_list[idx]

# 创建一个包含多个图的数据集
# 设定要生成的图数量 设定每个图的最大节点数量
dim_feature = 16
custom_dataset = CustomDataset(5, 30, dim_feature= dim_feature)

# 打印生成的图数据
# for i in range(len(custom_dataset)):
#     data = custom_dataset.get(i)
#     print(f"Graph {i + 1} data:", data)
#     print(data.x)
#
# for i, data in enumerate(custom_dataset):
#     print(f"enumerate Graph {i + 1} data:", data)


# 生成两个图数据
# graph_data_1 = CustomDataset.generate_random_graph(20)  # 生成包含 20 个节点的图数据
# graph_data_2 = CustomDataset.generate_random_graph(30)  # 生成包含 30 个节点的图数据
# print("Graph 1 data:", graph_data_1)
# print("Graph 2 data:", graph_data_2)

# edge_index = torch.tensor([[0, 0, 0, 1],
#                             [1, 2, 3, 2]], dtype=torch.long)
# x = torch.randn(4, 16)  # 4个节点，每个节点的特征向量大小为16
# data = Data(x=x, edge_index=edge_index)

data = custom_dataset

# 2. 定义 GAT 模型：
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATNodeRecommendation(nn.Module):
    def __init__(self, in_channels, out_channels, heads):
        super(GATNodeRecommendation, self).__init__()
        self.conv1 = GATConv(in_channels, out_channels, heads=heads)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        return F.log_softmax(x, dim=1)


# 3. 模型训练：
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GATNodeRecommendation(in_channels=16, out_channels=8, heads=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fun = F.nll_loss

def train(model, train_dataset, optimizer, loss_fun, epochs = 100):
    model.train()
    for epoch in range(epochs):
        loss_sum = 0.0
        for i, data in enumerate(train_dataset):
            optimizer.zero_grad()
            out = model(data)
            # loss = F.nll_loss(out, torch.tensor([0, 1, 2, 3]))  # 通过loss function指定节点
            loss = loss_fun(out, torch.tensor([i for _ in range(data.x.size(0))]))  # 通过loss function指定节点
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        print(f'Epoch [{epoch + 1}], Loss: {loss_sum/len(train_dataset)}')
    print('训练结束')

# train(model, custom_dataset, optimizer, loss_fun, 100)

# 4. 模型测试：
model.eval()
# with torch.no_grad():
#     logits = model(data)
#     # 获取前k个预测节点的索引
#     k = 3
#     top_k = logits.topk(k=k, dim=1).indices
#     print(f'Top {k} recommended nodes for the given nodes: {top_k}')

import sys
def beam_search(model, data, k, width, depth):
    # 初始化初始节点
    initial_nodes = [0]  # 这里以第一个节点作为初始节点
    paths = {tuple(initial_nodes): 1.0}  # 存储每个路径及其对应的概率，初始概率为 1.0

    for _ in range(depth):
        new_paths = {}
        for path, score in paths.items():
            node = path[-1]  # 当前路径的最后一个节点
            with torch.no_grad():
                logits = model(data)  # 获取预测值
                # probabilities = F.softmax(logits[node], dim=0)  # 计算节点的预测概率
                try:
                    probabilities = F.softmax(logits[node], dim=0)  # 计算节点的预测概率
                except IndexError as ie:
                    print(data, len(logits), node, path)
                    # sys.exit(-1)

            top_k_prob, top_k_indices = torch.topk(probabilities, width)  # 获取 top k 的概率和索引
            for i in range(width):
                new_path = tuple(list(path) + [top_k_indices[i].item()])  # 扩展路径
                new_paths[new_path] = score * top_k_prob[i].item()  # 更新路径和概率
        paths = dict(sorted(new_paths.items(), key=lambda item: item[1], reverse=True)[:width])  # 保留 top k 的路径
    top_k_paths = list(paths.keys())[:k]  # 选取前 k 个路径
    return top_k_paths


# 测试集数据# 定义你的测试集数据
test_data = CustomDataset.generate_random_graph(12)

# 选择一个节点作为初始节点，假设是第一个节点
initial_node = 0

# 使用 beam_search 获取前 k 个预测节点索引
top_k_paths = beam_search(model, test_data, k=5, width=2, depth=3)

# 输出前 k 个预测节点索引
print("Top k paths:")
for path in top_k_paths:
    print(path)

