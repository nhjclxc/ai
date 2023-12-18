#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/18 13:59
# Module    : test2_impl_vocabulary.py
# explain   :

import collections
import re
from d2l import torch as d2l

# 1. 读取文件

def read_lines(path):
    lines = []
    with open(path, 'r') as f:
        lines = f.readlines()
    # 将所有字符转化为a-z和空格' '
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


lines = read_lines('../data/timemachine.txt')
print(len(lines))
print(lines[0])
print(lines[10])

# 2. 词元化
def tokenize(lines, token='word'):  #@save
    tokens = []
    if token == 'word':
        return [line.split(sep=' ') for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('未知的分词标识')
    return tokens

tokens = tokenize(lines)
for i in range(11):
    print(tokens[i])


# 3. 词汇表
class Vocab:
    def __init__(self, tokens=None, min_feq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 保存索引到token的映射，列表
        self._idx2token = ['<unk>'] + reserved_tokens
        # 保存token到索引的映射，dict
        self._token2idx = {token: index for index, token in enumerate(self._idx2token)}

        # 获取词频率
        self._token_freqs = Vocab.counter(tokens, min_feq)

        # 构造
        for token, freq in self._token_freqs:
            self._idx2token.append(token)
            self._token2idx[token] = len(self._idx2token) - 1

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

vocab = Vocab(tokens)
print(list(vocab._token2idx.items())[:10])


for i in [0, 10]:
    print('文本:', tokens[i])
    print('索引:', vocab[tokens[i]])
    # print('索引:', vocab.__getitem__(tokens[i]))


# 整合所有功能
'''
在使用上述函数时，我们[将所有功能打包到load_corpus_time_machine函数中]， 该函数返回corpus（词元索引列表）和vocab（时光机器语料库的词表）。 我们在这里所做的改变是：
    为了简化后面章节中的训练，我们使用字符（而不是单词）实现文本词元化；
    时光机器数据集中的每个文本行不一定是一个句子或一个段落，还可能是一个单词，因此返回的corpus仅处理为单个列表，而不是使用多词元列表构成的一个列表。
'''
def load_corpus_time_machine(max_tokens=-1):  #@save
    """返回时光机器数据集的词元索引列表和词表"""
    lines = read_lines('../data/timemachine.txt')
    tokens = tokenize(lines, 'word')
    vocab = Vocab(tokens)
    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

corpus, vocab = load_corpus_time_machine()
print(len(corpus))
print(len(vocab))