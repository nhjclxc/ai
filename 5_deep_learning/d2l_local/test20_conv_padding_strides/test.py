#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/16 22:24
# Module    : test.py
# explain   :


import torch
from torch import nn


# 为了方便起见，我们定义了一个计算卷积层的函数。
# 此函数初始化卷积层权重，并对输入和输出提高和缩减相应的维数
def comp_conv2d(conv2d, X):
    # 这里的（1，1）表示批量大小和通道数都是1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # 省略前两个维度：批量大小和通道
    return Y.reshape(Y.shape[2:])

# 请注意，这里每边都填充了1行或1列，因此总共添加了2行或2列
# 这里设置padding=1的意思是在原始图像的上下左右各添加一行或一列，对于整个图像的话总共增加了两行和两列
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
X = torch.rand(size=(8, 8))
# 8-3+2+1=8
print(comp_conv2d(conv2d, X).shape)

# 当卷积核的高度和宽度不同时，我们可以[填充不同的高度和宽度]
conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
# (8-5+4+1, 8-3+2+1) = (8, 8)
print(comp_conv2d(conv2d, X).shape)

#[将高度和宽度的步幅设置为2]
# (8-3+2+1)/2=4
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
print(comp_conv2d(conv2d, X).shape)


# 同时设置不同的填充和步幅
# h = ⌊(8-3+0+3)/3⌋ = ⌊ 2.67 ⌋ =2（向下取整）
# w = ⌊(8-5+2+4)/4⌋ = ⌊ 2.25 ⌋ =2（向下取整）
conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
print(comp_conv2d(conv2d, X).shape)




