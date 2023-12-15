#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/14 20:35
# Module    : test02_TensorBoard.py
# explain   :TensorBoard的使用

from torch.utils.tensorboard import SummaryWriter
import torch

writer = SummaryWriter('train_log')

for x in range(100):
    writer.add_scalar('y=log(x)', torch.log(torch.tensor(x)), x)

writer.close()



