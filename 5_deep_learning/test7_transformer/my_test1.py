#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2024/1/4 21:34
# Module    : my_test1.py
# explain   :



import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        self.query = nn.Linear(embed_size, embed_size)  # 或者 nn.Linear(self.head_dim, self.head_dim)
        self.key = nn.Linear(embed_size, embed_size)  # 或者 nn.Linear(self.head_dim, self.head_dim)
        self.value = nn.Linear(embed_size, embed_size)  # 或者 nn.Linear(self.head_dim, self.head_dim)

        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, query, key, value, mask):
        batch_size = query.shape[0]

        # Split embedding into multiple heads and perform linear transformation
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)

        # Reshape Q, K, V to (batch_size, heads, seq_len, head_dim)
        Q = Q.view(batch_size, -1, self.heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.heads, self.head_dim).permute(0, 2, 1, 3)

        # Compute scores and scale
        scores = torch.matmul(Q, K.permute(0, 1, 3, 2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = F.softmax(scores, dim=-1)

        # Apply attention to value
        output = torch.matmul(attention, V)
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.heads * self.head_dim)

        return self.fc_out(output)

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads):
        super(TransformerBlock, self).__init__()
        self.multihead_attention = MultiHeadAttention(embed_size, heads)
        self.layer_norm1 = nn.LayerNorm(embed_size)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size)
        )
        self.layer_norm2 = nn.LayerNorm(embed_size)

    def forward(self, x, mask):
        attention = self.multihead_attention(x, x, x, mask)
        x = self.layer_norm1(x + attention)
        feedforward = self.feedforward(x)
        x = self.layer_norm2(x + feedforward)
        return x

class Transformer(nn.Module):
    def __init__(self, embed_size, heads, num_layers):
        super(Transformer, self).__init__()
        self.num_layers = num_layers
        self.transformer_blocks = nn.ModuleList([TransformerBlock(embed_size, heads) for _ in range(num_layers)])

    def forward(self, x, mask):
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)
        return x

# 创建一个Transformer模型
embed_size = 512
heads = 8
num_layers = 6
model = Transformer(embed_size, heads, num_layers)

# 定义输入和mask
input_sequence = torch.randn(10, 20, embed_size)  # (batch_size, sequence_length, embed_size)
mask = torch.ones(10, 20)  # 用于屏蔽填充token

# 将输入传入模型
output_sequence = model(input_sequence, mask)



