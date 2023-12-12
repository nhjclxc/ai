#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/12 15:51
# Module    : test01_softmax.py
# explain   : 从零实现softmax


import torch
from IPython import display
from d2l import torch as d2l

# 1. 数据加载
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 2. 初始化模型参数


