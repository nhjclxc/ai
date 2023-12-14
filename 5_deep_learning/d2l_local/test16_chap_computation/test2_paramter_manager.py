#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/14 13:27
# Module    : test2_paramter_manager.py
# explain   : 使用nn.Module创建的神经网络参数管理


import torch
from torch import nn

'''
# 注意：只有线性层才有参数，这个参数其实就是权重W和偏置b，像relu，softmax这种是没有参数的，不能访问到，更无法修改
# 后期如果要变量神经网络的每一层一定要判断该module是不是nn.Linear，即：if type(m) == nn.Linear:

'''
# nn.Linear(4, 8)索引为0，nn.ReLU()索引为1，nn.Linear(8, 1)索引为2
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
print(net(X))

print('\n\n [访问神经网络]')
print(net[0])
print(net[1])
print(net[2])

print('\n\n [参数访问]')
# 我们从已有模型中访问参数。 当通过Sequential类定义模型时， 我们可以通过索引来访问模型的任意层。 这就像模型是一个列表一样，每层的参数都在其属性中。
# state_dict()返回这个神经网络（也就是net[2]）的所有参数
net_layer2_param_dict = net[2].state_dict()
print(net_layer2_param_dict, type(net_layer2_param_dict))
print(net_layer2_param_dict['weight'], type(net_layer2_param_dict['weight']))
print(net_layer2_param_dict['bias'], type(net_layer2_param_dict['bias']))
'''
OrderedDict([('weight', tensor([[-0.1434,  0.1381,  0.2019, -0.1055, -0.3384,  0.0571,  0.3362,  0.2159]])), ('bias', tensor([0.1859]))])
    OrderedDict表示有序的字典，类型是<class 'collections.OrderedDict'>
    [('weight', ...), ('bias', ...)]表示这个有序字典集合里面有两个参数
        第一个参数是('weight', ...)：'weight'是参数名字，后面接着的就是这个参数的真正数据，也就是一个torch的张量
'''

print('\n\n [目标参数]')
# 每个参数都表示为参数类的一个实例。
print(type(net[2].weight))
print(net[2].weight)
print(net[2].weight.data)
print()
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)

# 由于现在还没有进行网络反向传播，因此下面的输出为True
print(net[2].weight.grad == None)
# 反向传播
# net[2].backward()
# print(net[2].weight.grad == None)


print('\n\n [一次性访问所有参数]')
print(net[0].named_parameters())
print(*[(name, param.shape, param.data) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])


print('\n\n [从嵌套块收集参数]')
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4), nn.ReLU())


def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block {i}', block1())
    return net


rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
print(rgnet(X))
print(rgnet)
'''
下面这个Sequential表示的就是rgnet里面的所有有序的神经网络
Sequential(
  #   (0): Sequential(表示的就是block2()
  (0): Sequential(
    # 下面的(block i)表示的就是block1()
    (block 0): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
    (block 1): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
    (block 2): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
    (block 3): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
  )
  # (1): Linear表示的就是：nn.Linear(4, 1)
  (1): Linear(in_features=4, out_features=1, bias=True)
)
'''
print(rgnet[0][1][0].bias.data)


print('\n\n\n 参数初始化')
print('\n [内置初始化]')
# 让我们首先调用内置的初始化器。 下面的代码将所有权重参数初始化为标准差为0.01的高斯随机变量， 且将偏置参数设置为0。
def init_normal(m):
    # 这个m表示神经网络的每一层的module
    # 只有线性层才有权重和偏置
    if type(m) == nn.Linear:
        # 初始化权重为正太分布
        nn.init.normal_(m.weight, mean=0, std=0.01)
        # nn.init.constant_(m.weight, 1)

        # 偏置为全0
        nn.init.zeros_(m.bias)
        # nn.init.zeros_(m.bias)

# apply方法传入的是一个lambda表达式，会对神经网络net里面的每一层module应用于该方法上
net.apply(init_normal)
print(net[0].weight.data[0])
print(net[0].bias.data[0])

# 为某个层module单独应用初始化权重
net[0].apply(init_normal)


print('\n\n [[参数绑定]]')
'''
有时我们希望在多个层间共享参数： 我们可以定义一个稠密层，然后使用它的参数来设置另一个层的参数。
'''
# 我们需要给共享层一个名称，以便可以引用它的参数
shared = nn.Linear(8, 8)
# 这个神经网络里面的两个隐藏层公用同一个神经网络，里面的参数是同一套，因为这两个层都是使用了同一个线性层shared
# 也就是同一个nn.Linear的实列，一个实列里面只有一套参数，这一套参数在前面的隐藏层可以被修改，在后面的隐藏层也可以被修改
# 也就是修改的参数只有一份，所以达到了共享参数的目的
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同一个对象，而不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])









