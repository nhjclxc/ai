#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/14 19:51
# Module    : test01_DataSet.py
# explain   : 构造数据集


import os

from PIL import Image
from torch.utils.data import Dataset

# 蜜蜂数据集：https://download.pytorch.org/tutorial/hymenoptera_data.zip

class MyDataSet(Dataset):

    def __init__(self, root_path, label):
        self._root_path = root_path
        self._label = label
        self._dataset_path = os.path.join(self._root_path, self._label)
        self._dataset = os.listdir(self._dataset_path)

    # 必须实现方法1：用于返回输入数据和标签对
    def __getitem__(self, idx):
        data_name = self._dataset[idx]
        data_path = os.path.join(self._dataset_path, data_name)
        img = Image.open(data_path)
        return (img, self._label)

    # 必须实现方法2
    def __len__(self):
        return len(self._dataset)

root_path = '../data/hymenoptera_data/train'
label_ants = 'ants'
ants_dataset = MyDataSet(root_path, label_ants)
print(type(MyDataSet))
print(ants_dataset)
print(ants_dataset.__len__())
input, label = ants_dataset[0]
print(input)
print(label)
# input.show()

label_bees = 'bees'
bees_dataset = MyDataSet(root_path, label_bees)
print(bees_dataset.__len__())

dataset = ants_dataset + bees_dataset
print(dataset.__len__())


