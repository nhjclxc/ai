#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/29 10:55
# Module    : test3.py
# explain   :

import dgl
import torch
from torch_geometric.data import Data

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

# 字典
vocab = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
            'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']
embedding_size = 16

# 生成一个简单的图
x = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)  # 示例的节点索引
edge_index = torch.tensor([ [0, 1, 1, 2, 2, 4],
                            [1, 0, 2, 3, 4, 3]], dtype=torch.long)  # 示例的边索引
edge_attr = torch.tensor([  [0.1, 0.2],
                            [0.3, 0.4],
                            [0.3, 0.4],
                            [0.3, 0.4],
                            [0.3, 0.4],
                            [0.5, 0.6]], dtype=torch.float)  # 示例的边属性
data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
print(data)
node_embeds = torch.nn.Embedding(len(vocab), embedding_size)
print(node_embeds(torch.tensor([1])))
print(node_embeds(torch.tensor([1,2])))



# 定义 GNN 模型
class GATModel(nn.Module):
    def __init__(self, in_channels, out_channels, heads):
        super(GATModel, self).__init__()
        torch.manual_seed(999)

        # 一个 GAT 层的输出维度为 out_channels * heads，其中 out_channels 是指定的输出特征维度。
        # 即输入数据经过GATConv的使出维度并不是out_channels，而是out_channels*heads，
        # 那么下一层接收的时候就要定义out_channels*heads，而不是像GCN那样简单的拿out_channels
        # (n*16)*
        self.conv1 = GATConv(in_channels, 32, heads=2, dropout=0.1)  # 输入特征维度16，输出特征维度8
        # 64*len(vocab)
        self.conv2 = GATConv(32 * 2, out_channels, heads=1, dropout=0.1)  # 输入特征维度8，输出特征维度4

    # def forward(self, x, edge_index, edge_attr):
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # x是一个n*16的张量,n是当前图的节点个数,16是每一个节点的嵌入维度
        # 输出x = (n*16) * (16*32*2) = n*64
        x = self.conv1(x, edge_index, edge_attr)
        # n * 32
        x = nn.ReLU()(x)
        # 输出: (n*64) * (64*len(vocab)) = n*len(vocab)
        x = self.conv2(x, edge_index, edge_attr)
        return x

# 初始化模型
hidden_dim = 32
output_dim = len(vocab)
model = GATModel(embedding_size, hidden_dim, output_dim)

# 定义一个简单的损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 开始训练
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    logits = model(data)
    loss = criterion(logits, train_labels)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')


'''
def beam_search(model, g, start_node, vocab, k, beam_width):
    # 将起始节点转换为词汇表索引
    start_node_idx = vocab.index(start_node) if start_node in vocab else None

    if start_node_idx is not None:
        # 初始化beam
        beam = [([start_node_idx], 0)]  # 初始的beam为起始节点索引和初始概率值0

        for _ in range(k):  # 重复k次搜索
            new_beam = []
            for hypothesis, score in beam:
                # 取出当前假设序列的最后一个节点
                current_node = hypothesis[-1]

                # 获取当前节点的概率分布
                node_scores = model(g, g.ndata['feat'])

                # 获取当前节点的top beam_width个节点索引及对应的分数
                top_scores, top_indices = torch.topk(node_scores[current_node], beam_width)

                # 对每个top节点进行扩展
                for i in range(beam_width):
                    new_hypothesis = hypothesis + [top_indices[i].item()]
                    new_score = score + top_scores[i].item()
                    new_beam.append((new_hypothesis, new_score))

            # 对新的beam进行排序并选择top k个
            beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:k]

        # 将beam中的节点索引转换回节点值
        top_k_nodes = [[vocab[idx] for idx in hyp] for hyp, _ in beam]
        return top_k_nodes
    else:
        return []
'''
import torch
import torch

def beam_search(model, start_node, vocab, k, beam_width):
    # 将起始节点转换为词汇表索引
    start_node_idx = vocab.index(start_node) if start_node in vocab else None

    if start_node_idx is not None:
        beam = [([start_node_idx], 0)]

        for _ in range(k):
            new_beam = []
            for hypothesis, score in beam:
                current_node = hypothesis[-1]

                # 获取当前节点的概率分布
                node_scores = model(graph, graph.ndata['feat'])
                current_scores = node_scores[current_node]

                # 获取整个词汇表的 top beam_width 个节点索引及对应的分数
                top_scores, top_indices = torch.topk(current_scores, len(vocab))

                for i in range(beam_width):
                    new_hypothesis = hypothesis + [top_indices[i].item()]
                    new_score = score + top_scores[i].item()
                    new_beam.append((new_hypothesis, new_score))

            beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:k]

        top_k_nodes = [[vocab[idx] for idx in hyp] for hyp, _ in beam]
        return top_k_nodes
    else:
        return []

# 假设有一个训练好的模型 model，一张图 g，节点词汇表 vocab
# start_node 是起始节点，k 是要返回的 top-k 节点数
start_node = 'A'
k = 2  # 假设要返回 top-5 的节点

# 定义边列表



# top_k_nodes = search_top_k_nodes(model, g, start_node, vocab, k)
beam_width = 3
top_k_nodes = beam_search(model, start_node, vocab, k, beam_width)
print(top_k_nodes)
