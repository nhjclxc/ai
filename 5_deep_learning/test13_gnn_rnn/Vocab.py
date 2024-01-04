#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/26 10:59
# Module    : Vocab.py
# explain   : 构建字典 词汇表

import torch
import collections

'''
将labels可以类比为接口api，对所有的api接口进行编码，以获得每一个api的索引，即获得了api集合的词汇表Vocab
'''


class Vocab:
    def __init__(self, tokens=None, min_feq=0, embedding_size=16, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 保存索引到token的映射，列表
        self._idx2token = ['<unk>'] + reserved_tokens
        # 保存token到索引的映射，dict
        self._token2idx = {token: index for index, token in enumerate(self._idx2token)}

        # 获取词频率，出现频率越高的越在前面（索引越小）
        self._token_freqs = Vocab.counter(tokens, min_feq)

        # 构造
        for token, freq in self._token_freqs:
            self._idx2token.append(token)
            self._token2idx[token] = len(self._idx2token) - 1

        # 节点嵌入
        self.node_embeds = torch.nn.Embedding(len(self._idx2token), embedding_size)


    def get_embedding_by_indices(self, indices):
        if isinstance(indices, (list,tuple)):
            indices = torch.tensor(indices)
        return self.node_embeds(indices)

    def get_embedding_by_tokens(self, tokens):
        indices = self.__getitem__(tokens)
        return self.get_embedding_by_indices(torch.tensor(indices))


    def __len__(self):
        return len(self._idx2token)

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    def __getitem__(self, item):
        """ 通过token获取所有索引列表 """
        if not isinstance(item, (list,tuple)):
            # item不以列表形式进来，直接使用self的字典返回
            return self._token2idx.get(item, self.unk)
        return [self.__getitem__(token) for token in item]

    def to_tokens(self, indices):
        """ 通过indices获取对应的token列表 """
        """ 通过token获取所有索引列表 """
        if not isinstance(indices, (list,tuple)):
            # item不以列表形式进来，直接使用self的字典返回
            return self._idx2token[indices]
        return [self.to_tokens(i) for i in indices]

    def check_index(self, index):
        return True if index < len(self._idx2token) else False

    @staticmethod
    def counter(tokens, min_feq = 0):
        """ 词频率排序 """
        if len(tokens) == 0 or isinstance(tokens[0], (list, tuple)):
            tokens = [token for line in tokens for token in line if token != '']
        # 统计每个字符出现的次数
        counter = collections.Counter(tokens)
        # 排序
        sorted_counter = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # 剔除小于最小频率的，并返回
        return [(token, freq) for token, freq in sorted_counter if freq >= min_feq]


if __name__ == '__main__':
    # 测试
    tokens = ['cat', 'dog', 'mouse', 'mouse', 'mouse', 'mouse', 'mouse', 'cat', 'dog', 'dog']
    vocab = Vocab(tokens)
    print(list(vocab._token2idx.items()))

    for i, v in enumerate(list(vocab._token2idx.items())):
        print(i, v)

    length = len(vocab._token2idx.items())
    for i in range(length):
        token = vocab.to_tokens([i])
        print(i, ' 文本:', token, '索引:', vocab[token])

    print(len(vocab._token2idx.items()))
    print(len(vocab))