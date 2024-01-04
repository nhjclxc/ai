#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/29 19:07
# Module    : test2.py
# explain   :
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv

class GATEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GATEncoder, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=4)
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=4)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

class AttRNNDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AttRNNDecoder, self).__init__()
        self.rnn = nn.GRUCell(input_dim, hidden_dim)
        self.attn = nn.Linear(hidden_dim, 1)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        hidden_dim = x.size(-1)

        c = torch.zeros((batch_size, hidden_dim)).to(x.device)
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            h = self.rnn(x_t, c)

            attn_weights = torch.softmax(self.attn(h), dim=0)
            context = torch.sum(h * attn_weights, dim=0)

            output = self.linear(context)
            outputs.append(output)

        return torch.stack(outputs, dim=1)


# 随机生成图数据
x = torch.randn((5, 10))  # 特征向量维度为10
edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
data = Data(x=x, edge_index=edge_index)

# 随机生成序列数据
# seq_data = [torch.randn((5, 10)) for _ in range(5)]  # 序列长度为5，特征向量维度为10
seq_data = [torch.unsqueeze(torch.randn((5, 10)), dim=0) for _ in range(5)]  # 添加一个维度

# 初始化编码器和解码器
encoder = GATEncoder(input_dim=10, hidden_dim=16)
decoder = AttRNNDecoder(input_dim=16, hidden_dim=16, output_dim=10)

# 训练过程
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)
for _ in range(100):
    optimizer.zero_grad()

    encoded = encoder(data.x, data.edge_index)
    decoded = decoder(encoded)

    loss = F.mse_loss(decoded, data.x)
    loss.backward()
    optimizer.step()

# 推断过程
with torch.no_grad():
    encoded = encoder(data.x, data.edge_index)
    reconstructed = decoder(encoded)




