#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2024/6/1 21:35
# Module    : test05_nnModule.py
# explain   :

# https://pytorch.org/docs/stable/nn.html
# https://pytorch.org/docs/stable/generated/torch.nn.Module.html

import torch
from torch import nn
import torch.nn.functional as F


# 第一自定义的神经网络模块
class MyModule(nn.Module):

    def __init__(self, weight):
        super(MyModule, self).__init__()
        self.weight = weight

    def forward(self, input):
        # 使用input * self.weight模拟神经网络操作
        return input * self.weight

def test_MyModule():
    myModule = MyModule(2)
    input = 1
    for _ in range(5):
        print('input', input)
        output = myModule(input)
        print('output', output)
        input = output
        print('------------------')
# test_MyModule()

def test_conv2d():
    input = torch.tensor(  [[1,2,0,3,1],
                            [0,1,2,3,1],
                            [1,2,1,0,0],
                            [5,2,3,1,1],
                            [2,1,0,1,1]])
    # batch_size, channels, weigth, width
    input = torch.reshape(input, (1, 1, 5, 5))
    k = torch.tensor(  [[1,2,1],
                        [0,1,0],
                        [2,1,0]])
    k = torch.reshape(k, (1, 1, 3, 3))
    # stride滑动窗口移动步长
    output = F.conv2d(input=input, weight=k, stride=1)
    print(output)
    # batch_size, channels, weigth, width
    # tensor([[[[10, 12, 12],
    #           [18, 16, 16],
    #           [13,  9,  3]]]])

    print(output[0])
    # channels, weigth, width
    # tensor([[[10, 12, 12],
    #          [18, 16, 16],
    #          [13,  9,  3]]])

    print(output[0][0])
    # weigth, width
    # tensor([[10, 12, 12],
    #         [18, 16, 16],
    #         [13,  9,  3]])

    print(output[0][0][0])
    # width
    # tensor([10, 12, 12])
# test_conv2d()

def test_conv2d_2():
    input = torch.tensor(  [[1,2,0,3,1],
                            [0,1,2,3,1],
                            [1,2,1,0,0],
                            [5,2,3,1,1],
                            [2,1,0,1,1]])
    # batch_size, channels, weigth, width
    input = torch.reshape(input, (1, 1, 5, 5))
    k = torch.tensor(  [[1,2,1],
                        [0,1,0],
                        [2,1,0]])
    k = torch.reshape(k, (1, 1, 3, 3))
    # stride滑动窗口移动步长
    output = F.conv2d(input=input, weight=k, stride=1, padding=1)
    print(output)

test_conv2d_2()


