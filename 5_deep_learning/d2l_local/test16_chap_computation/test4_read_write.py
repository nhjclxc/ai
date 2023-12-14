#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/14 16:53
# Module    : test4_read_write.py
# explain   :

import torch
from torch import nn
from torch.nn import functional as F

x = torch.arange(4)
print(x)
# 使用save保存参数
torch.save(x, 'x-file.params')

# 使用load读取参数
load_x = torch.load('x-file.params')
print(load_x)


# 保存模型参数
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
# 训练完成后保存参数
torch.save(net.state_dict(), 'mlp.params')

# 另一个模型加载参数
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
print(clone.eval())

