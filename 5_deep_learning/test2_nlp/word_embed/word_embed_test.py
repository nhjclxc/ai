#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/21 18:25
# Module    : word_embed_test.py
# explain   :


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


labels = [
    "ws",
    "ws",
    "wddf",
    "ggg",
    "hh",
    "hh",
    "ggg"
]

# 构建字典表示，使用不同的标签作为键，值为该标签第一次出现的索引值
label_dict = {label: index for index, label in enumerate(sorted(set(labels)))}

print(label_dict)



from collections import Counter

labels = [
    "ws",
    "ws",
    "wddf",
    "ggg",
    "hh",
    "hh",
    "ggg"
]

# 计算标签出现的频率
label_freq = Counter(labels)

# 筛选出现频率大于一次的标签，并按照频率从大到小排序
filtered_labels = {label: index for index, (label, freq) in enumerate(label_freq.most_common()) if freq > 1}

print(filtered_labels)
# 将所有键放入列表中
keys_list = list(filtered_labels.keys())

print(keys_list)




word_to_ix = {"hello": 0, "world": 1}
# 2 words in vocab，词汇表大小为2, 5 dimensional embeddings，嵌入的维度为5
embeds = nn.Embedding(2, 5)
lookup_tensor = torch.tensor([word_to_ix["hello"]], dtype=torch.long)
print(f'lookup_tensor = {lookup_tensor}')
hello_embed = embeds(lookup_tensor)
print(f'hello_embed = {hello_embed}')

lookup_tensor2 = torch.tensor([0,1], dtype=torch.long)
print(embeds(lookup_tensor2))
