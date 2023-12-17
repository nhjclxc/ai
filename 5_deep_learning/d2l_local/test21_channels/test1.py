#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/16 22:47
# Module    : test1.py
# explain   :


import torch

'''
什么是通道？

什么是输入通道？
    也称为深度（Depth），表示输入数据的通道数。对于彩色图像来说，通道数通常是 RGB 三个通道（红、绿、蓝）。对于灰度图像，通道数为 1。这个维度表示输入数据的不同特征或信息层。

什么是输出通道？
    在卷积层中，卷积核的数量决定了输出通道数。每个卷积核都会产生一个输出通道。输出通道的数量可以看作是卷积层学习到的特征数量，每个通道对应不同的特征。


'''
import torch
from d2l import torch as d2l


def test1():
    x1 = torch.tensor([1, 2, 3])
    x2 = torch.tensor([4, 5, 6])
    x3 = torch.tensor([7, 8, 9])
    print(torch.cat((x1, x2), dim=0))
    print(torch.cat((x1, x2, x3), dim=0))
    print(torch.stack((x1, x2), dim=0))
    print(torch.stack((x1, x2, x3), dim=0))

    print('\n\n')
    x1 = torch.tensor([[1, 2, 3]])
    x2 = torch.tensor([[4, 5, 6]])
    x3 = torch.tensor([[7, 8, 9]])
    print(torch.cat((x1, x2), dim=0))
    print(torch.cat((x1, x2, x3), dim=0))
    print(torch.stack((x1, x2), dim=0))
    print(torch.stack((x1, x2, x3), dim=0))
    pass


def corr2d(X, K):  # @save
    """计算二维互相关运算"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y

def conv2d_mult_in(X, K):
    Y_i = torch.zeros(size=(X.shape[1] - K.shape[1] + 1,X.shape[2] - K.shape[2] + 1))
    for i in range(X.shape[0]):
        Y_i += corr2d(X[i, :, :], K[i, :, :])
    return Y_i

def corr2d_multi_in2(X, K):
    # 先遍历“X”和“K”的第0个维度（通道维度），再把它们加在一起
    return sum(corr2d(x, k) for x, k in zip(X, K))
def test2():
    # 卷积核形状时：Ci * Kh * Kw
    '''多输入通道
       当输入通道c = 1时，输入数据的张量的卷积核为h*w
       当输入通道c > 1时，输入数据的张量的卷积核为c*h*w。
        由于输入和卷积核都有c个通道，我们可以对每个通道输入的二维张量和卷积核的二维张量进行互相关运算，再对通道求和（将每个通道的卷积结果相加）得到二维张量。这是多通道输入和多输入通道卷积核之间进行二维互相关运算的结果。
    '''

    X = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                      [[1, 2,3], [ 4, 5,6], [7, 8,9]]])
    K = torch.tensor([[[0, 1], [2, 3]],
                      [[1, 2], [3, 4]]])

    print(conv2d_mult_in(X, K))




    pass


# 多输出通道
def test3():
    # 卷积核形状时：Co * Ci * Kh * Kw

    X = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                      [[1, 2,3], [ 4, 5,6], [7, 8,9]]])
    K = torch.tensor([[[0, 1], [2, 3]],
                      [[1, 2], [3, 4]]])

    print(conv2d_mult_in(X, K))


    pass


if __name__ == '__main__':

    # test1()

    # test2()

    test3()


    pass

