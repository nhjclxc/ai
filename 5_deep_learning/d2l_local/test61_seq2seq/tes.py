#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/20 10:41
# Module    : tes.py
# explain   :
import torch
import torch.nn as nn


def test1():
    # 定义一个嵌入层，词汇表大小为 10，嵌入维度为 5
    embedding_layer = nn.Embedding(10, 5)

    print(embedding_layer(torch.LongTensor([0])))
    print(embedding_layer(torch.LongTensor([0, 1])))
    print(embedding_layer(torch.LongTensor([1])))
    print()

    print(embedding_layer(torch.LongTensor([2])))
    print(embedding_layer(torch.LongTensor([3])))
    print(embedding_layer(torch.LongTensor([4])))
    print(embedding_layer(torch.LongTensor([5])))
    print(embedding_layer(torch.LongTensor([6])))
    print(embedding_layer(torch.LongTensor([7])))
    print(embedding_layer(torch.LongTensor([8])))
    print(embedding_layer(torch.LongTensor([9])))
    # IndexError: index out of range in self
    # print(embedding_layer(torch.LongTensor([10])))
    # print(embedding_layer(torch.LongTensor([11])))

    pass


def test2():
    X = torch.tensor([[1,2,3],[4,5,6]])
    print(X)
    X = X.permute(1, 0)  # 转化输入张量X的维度,对调原来的0,1维度
    print(X)
    pass


def test3():
    X = torch.tensor([[1,2,3],[4,5,6]])

    print(X)

    print(X.repeat(1,0))
    print(X.repeat(1,1))
    print(X.repeat(2,1))
    print(X.repeat(2,3))
    print(X.repeat(2,3,2))

    pass


def test4():
    X = torch.tensor(   [[1, 2,3],
                        [4,5,6]])
    Y = torch.tensor(   [[11,22,33],
                        [44,55,66]])


    print(torch.cat((X, Y), 0))
    print(torch.cat((X, Y), 1))

    pass


if __name__ == '__main__':
    # test1()

    # test2()

    # test3()

    # test4()

    # X = torch.tensor([[1, 2, 3], [4, 5, 6]])
    # print(X.size(1))

    X = torch.tensor([1, 2, 3])
    Y = torch.tensor([11, 22, 33])
    # print(torch.cat((X, Y), dim=0))
    X_Y = torch.stack((X, Y), dim=0)
    tensor([1])
    tensor([3])

    pass

