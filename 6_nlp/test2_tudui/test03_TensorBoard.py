#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2024/5/30 21:48
# Module    : test03_TensorBoard.py
# explain   : TensorBoard的使用

from torch.utils.tensorboard import SummaryWriter

import torch

from PIL import Image
import numpy as np


def test_add_scalar():
    # train_log表示的是日志文件输出位置
    writer = SummaryWriter('logs')

    for x in range(10):
        # 标题, y值, x值
        # writer.add_scalar("y=x", x, x)
        writer.add_scalar("y=log(x)", torch.log(torch.tensor(x)), x)

    writer.close()

    # 查看
    # 在安装了torch的环境下,进入当前目录输入: tensorboard --logdir=logs
    # 一般地址是:http://localhost:6006/#timeseries

    # tensorboard add_scalar的作用是用于后期输出训练过程中的loss变化,这样就可以直观的看出损失了

# test_add_scalar()

def test_add_image():
    # train_log表示的是日志文件输出位置
    writer = SummaryWriter('logs')
    img_path = '../data/hymenoptera_data/train/ants/0013035.jpg'
    img = Image.open(img_path)
    print(type(img))
    np_arr = np.array(img)
    print(type(np_arr))


    writer.add_image('img', np_arr, 1, dataformats='HWC')

    writer.close()

test_add_image()