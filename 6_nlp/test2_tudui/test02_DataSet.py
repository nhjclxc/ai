#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2024/5/30 21:29
# Module    : test02_DataSet.py
# explain   : 数据集的构造

import os

from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class MyDataSet(Dataset):
    """
        继承torch的Dataset必须实现两个方法
            __getitem__和__len__
    """
    def __init__(self, root_dir, label):
        self.root_dir = root_dir
        self.label = label
        self.label_dir = os.path.join(self.root_dir, self.label)
        # self.dataset构成的是一个关于文件名的list，要想获取对应的实际文件必须加上其路径才可以
        self.dataset = os.listdir(self.label_dir)

    def __getitem__(self, idx):
        # 获取对应所有的文件名
        dataname = self.dataset[idx]
        # 读取文件
        data_dir = os.path.join(self.label_dir, dataname)
        # 读取数据
        img = Image.open(data_dir)
        # 构造返回数据集
        return (img, self.label)

    def __len__(self):
        return len(self.dataset)

def test_dataset1():
    root_dir = '../data/hymenoptera_data/train'
    # 蚂蚁数据集
    label = 'ants'
    ant_dataset = MyDataSet(root_dir, label)
    (img, label) = ant_dataset[0]
    # img.show()
    print(label)
    print(len(ant_dataset))

    # 蜜蜂数据集
    label = 'bees'
    bee_dataset = MyDataSet(root_dir, label)
    (img, label) = bee_dataset[0]
    # img.show()
    print(label)
    print(len(bee_dataset))

    # 将两个数据集相加构造整个训练数据集
    train_dataset = ant_dataset + bee_dataset
    print(len(train_dataset))
# test_dataset1()


import torchvision
def test_torchvsion_dataset():
    # 图片数据使用torchvision来加载
    # https://pytorch.org/vision/stable/datasets.html
    train_dataset = torchvision.datasets.CIFAR10(root = '../data', train=True, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root = '../data', train=False, download=True)
    img, label = test_dataset[0]
    print(img, label)
    print(test_dataset.classes)
    img.show()
    pass

# test_torchvsion_dataset()



# 构造好数据集后将数据集DataSet应用在Dataloader上面即可
# https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
class MyDataLoader(DataLoader):

    pass


test_dataset = torchvision.datasets.CIFAR10(root = '../data', train=False, download=True,
                                            transform=torchvision.transforms.ToTensor())
print('len(test_dataset)', len(test_dataset))
img, label = test_dataset[0]
# print(img)
print(label)

test_dataloader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=True, num_workers=0, drop_last=False)
print('len(test_dataloader)', len(test_dataloader))
# img1, label1 = test_dataloader.dataset[0]
# # print(img1)
# print(label1)
# print(img1 is img)
for data in test_dataloader:
    img1, label1 = data
    # torch.Size([4, 3, 32, 32])
    # 4表示有四个数据
    # 3表示每一张图片都是三通道的
    # 32，32分别表示
    print(img1.shape)
    # tensor([4, 9, 0, 9])
    # 4, 9, 0, 9表示上面img对应的图片数据的标签
    print(label1)
    break
