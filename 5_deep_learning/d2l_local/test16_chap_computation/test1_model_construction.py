#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/14 10:49
# Module    : test1_model_construction.py
# explain   :


import torch
from torch import nn
from torch.nn import functional as F
'''
自定义类继承torch的nn.Module，实现神经网络

pytorch的一些知识：http://www.feiguyunai.com/index.php/2019/09/11/pytorch-char03/
'''


# 1.定义一个两层的线性网络
print('1.定义一个两层的线性网络')
# 下面的代码包含一个具有256个单元和ReLU激活函数的全连接隐藏层， 然后是一个具有10个隐藏单元且不带激活函数的全连接输出层。
'''
nn.Linear(in_features, out_features, bias)表示神经网络的全连接层，
    in_features：指的是输入的二维张量的大小，即输入的[batch_size, size]中的size。
    out_features：指的是输出的二维张量的大小，即输出的二维张量的形状为[batch_size，output_size]，当然，它也代表了该全连接层的神经元个数。
    bias：表示是不是要偏置项
    从输入输出的张量的shape角度来理解，相当于一个输入为[batch_size, in_features]的张量变换成了[batch_size, out_features]的输出张量。
原文链接：https://blog.csdn.net/qq_42079689/article/details/102873766
'''
'''
是 PyTorch 中的一个容器，用于按顺序组织神经网络的层。它允许用户按照顺序定义网络的结构，将各种层组合起来以构建神经网络模型
'''
net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
X = torch.rand(2, 20)
print(net(X))


# 2. [自定义块]
print('2. [自定义块]')
'''
每个块必须提供的基本功能。
    将输入数据作为其前向传播函数的参数。
    通过前向传播函数来生成输出。请注意，输出的形状可能与输入的形状不同。例如，我们上面模型中的第一个全连接的层接收一个20维的输入，但是返回一个维度为256的输出。
    计算其输出关于输入的梯度，可通过其反向传播函数进行访问。通常这是自动发生的。
    存储和访问前向传播计算所需的参数。
    根据需要初始化模型参数。
'''
class MLP(nn.Module):
    '''
    含一个多层感知机，其具有256个隐藏单元的隐藏层和一个10维输出层。
    '''
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self):
        # 调用MLP的父类Module的构造函数来执行必要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params（稍后将介绍）
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隐藏层
        self.out = nn.Linear(256, 10)  # 输出层

    # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
    def forward(self, X):
        # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块中定义。
        return self.out(F.relu(self.hidden(X)))

net = MLP()
print(net(X))

# 3. [顺序块]
print('3. [顺序块]')
'''
Sequential的设计是为了把其他模块串起来。 为了构建我们自己的简化的MySequential， 我们只需要定义两个关键函数：
    一种将块逐个追加到列表中的函数；
    一种前向传播函数，用于将输入按追加块的顺序传递给块组成的“链条”。
'''
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # 这里，module是Module子类的一个实例。我们把它保存在'Module'类的成员
            # 变量_modules中。_module的类型是OrderedDict（有序字典：你传参的时候是怎么样的顺序，nn.Module就会把这个网络构造成对应的顺序）
            # _modules的主要优点是： 在模块的参数初始化过程中， 系统知道在_modules字典中查找需要初始化参数的子块。
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        # 当MySequential的前向传播函数被调用时， 每个添加的块都按照它们被添加的顺序执行。
        for block in self._modules.values():
            # 'block(X)'表示在本层计算输出，而使用'X = '将本层的输出又更新到变量X里面，随后下一趟循环进来的时候的X就是被上一层网络更新过后的输出了
            X = block(X)
        return X

net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
print(net(X))


# 4. [在前向传播函数中执行代码]
print('4. [在前向传播函数中执行代码]')
'''
有时我们可能希望合并既不是上一层的结果也不是可更新参数的项， 我们称之为常数参数（constant parameter）。 
例如，我们需要一个计算函数 f(X,W)=c*W.T*x的层， 其中X是输入， w是参数，c是某个在优化过程中没有更新的指定常量。 
因此我们实现了一个FixedHiddenMLP类，
'''
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随机权重参数。因此其在训练期间保持不变
        # 随机权重为20*20，每一行有20个权重表示一个输入有20个特征，总共有20行表示一个批量有20个输入
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        # 接收20个输入，有20个输出
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # 使用创建的常量参数以及relu和mm函数
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复用全连接层。这相当于两个全连接层共享参数
        X = self.linear(X)
        # 控制流
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()

net = FixedHiddenMLP()
print(net(X))

class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
print(chimera(X))

