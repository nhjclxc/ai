#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2024/5/30 22:19
# Module    : test04_Transforms.py
# explain   : 数据类型转化器
from PIL import Image
from torchvision import transforms

# 输入
# 输出
# 作用


# 将一张图片数据转化为tensor类型
img_path = '../data/hymenoptera_data/train/ants/0013035.jpg'
img = Image.open(img_path)
# <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=768x512 at 0x1C7FFA35840>
print(img)

def test1(img):
    # ToTensor

    # 转化
    toTensor = transforms.ToTensor()
    tensor = toTensor(img)
    print(tensor)
    print(tensor.shape) # torch.Size([3, 512, 768])
    print(type(tensor)) # <class 'torch.Tensor'>

    # print(transforms.ToTensor()(img))
    pass

# test1(img)

def test2(img):
    # Resize

    # (高height, 宽width)
    resize = transforms.Resize((256, 512))

    tensor = transforms.ToTensor()(img)
    print('tensor.shape', tensor.shape)
    resize_tensor = resize(tensor)
    print(type(resize_tensor))
    print('resize_tensor.shape', resize_tensor.shape)


    pass

test2(img)