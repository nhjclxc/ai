#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2024/6/1 21:35
# Module    : test06_nnModuleTest.py
# explain   :

# https://pytorch.org/docs/stable/nn.html
# https://pytorch.org/docs/stable/generated/torch.nn.Module.html

import torch
from torch import nn
import torch.nn.functional as F


# 第一自定义的神经网络模块
class MyModuleTest(nn.Module):

    def __init__(self, weight):
        super(MyModuleTest, self).__init__()
        self.weight = weight
        self.conv1 = torch.nn.Conv2d(3, 32, 3, 1, 1)
        self.relu1=torch.nn.ReLU()
        self.max_pooling1=torch.nn.MaxPool2d(2,1)
        self.dense1 = torch.nn.Linear(32 * 3 * 3, 128)
        # 通过Sequential来包装层
        self.module_block = nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 1)
        )

        self.module_block.add_module("module_block_dense", torch.nn.Linear(32 * 3 * 3, 128))


    def forward(self, input):
        # 使用input * self.weight模拟神经网络操作
        return input * self.weight

def test_MyModule():
    myModuleTest = MyModuleTest(2)
    print(myModuleTest)
    print(myModuleTest.module_block[1])
    print(myModuleTest.module_block[2])

    # for module in myModuleTest.children():
    #     print(module)
    for module in myModuleTest.modules():
        print(module)
        print('===============================')

    # input = 1
    # for _ in range(5):
    #     print('input', input)
    #     output = myModuleTest(input)
    #     print('output', output)
    #     input = output
    #     print('------------------')

test_MyModule()

print("你是谁？？？".center(100,"-"))