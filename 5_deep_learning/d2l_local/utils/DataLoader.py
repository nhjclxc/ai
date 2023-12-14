#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/14 9:49
# Module    : DataLoader.py
# explain   : 返回和torch.utils.data.DataLoader一样的数据加载器


import torch
from torch.utils.data import DataLoader, Dataset
import random
import string

x = 10
assert x == 5, "x 不等于 5"


class CustomDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

def create_custom_dataloader(inputs, labels, batch_size=1, shuffle=False):
    dataset = CustomDataset(inputs, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# 生成随机的 inputs 和 labels， 其中inputs 8个(1,8)的张量，labels 是8个对应的文字标签
inputs = [torch.rand(1, 8) for _ in range(8)]
labels = [''.join(random.choices(string.ascii_lowercase, k=5)) for _ in range(8)]

# 创建自定义的 DataLoader
# 总共有8个数据，生成两个批量，每一个批量是4个数据
custom_dataloader = create_custom_dataloader(inputs, labels, batch_size=4, shuffle=True)

# 遍历 custom_dataloader
for batch_inputs, batch_labels in custom_dataloader:
    print("Batch Inputs:", batch_inputs)
    print("Batch Labels:", batch_labels)


