#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/14 16:34
# Module    : test5_.py
# explain   :


import torch
import torch.nn.functional as F
from torch import nn

# 自定义一个层实现：y = W*X+b
class MyTestLinear(nn.Module):
    def __init__(self):
        super(MyTestLinear, self).__init__()
        # 初始化一个(3,3)的权重向量
        # self.weight = nn.Parameter(torch.tensor([[1,2,3],[4,5,6],[7,8,9]], dtype=torch.float), requires_grad=True)
        # self.bias = nn.Parameter(torch.tensor([1,2,3], dtype=torch.float))

        self.weight = nn.Parameter(torch.tensor([[1,0,0],[0,1,0],[0,0,1]], dtype=torch.float), requires_grad=True)
        self.bias = nn.Parameter(torch.ones(1,3, dtype=torch.float))

    def forward(self, X):
        return torch.mv(self.weight, X)+self.bias

net = MyTestLinear()
print(net(torch.tensor([1,2,3], dtype=torch.float)))