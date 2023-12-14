#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/14 15:48
# Module    : test3_custom-layer.py
# explain   :




import torch
import torch.nn.functional as F
from torch import nn

print('\n\n 不带参数的层')
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean() #均值

layer = CenteredLayer()
# 继承自 nn.Module 的类需要实现 forward 方法来定义模型的前向传播过程。如果没有显式地定义 forward 方法，会默认调用父类 nn.Module 中的 forward 方法。
print(layer(torch.FloatTensor([1, 2, 3, 4, 5])))
print(layer.forward(torch.FloatTensor([1, 2, 3, 4, 5])))


# [将层作为组件合并到更复杂的模型中]
# 这个神经网络先经过一个线性层在经过一个自定义层
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
Y = net(torch.rand(4, 8))
print(Y.mean())

net = nn.Sequential(nn.Linear(8, 128), nn.ReLU(), CenteredLayer())
Y = net(torch.rand(4, 8))
print(Y.mean())


print('\n\n [带参数的层]')
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        # w.shape=(in_units,units)，有in_units个输入特征，有units个输出特征
        self.weight = nn.Parameter(torch.randn(in_units, units))
        # 有units个偏置项，偏置项的个数和输出特征个数相同
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
linear = MyLinear(5, 3)
print(linear.weight)

print(linear(torch.rand(2, 5)))

net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
print(net(torch.rand(2, 64)))








