#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/2 17:40
# Module    : test1.py
# explain   :

# 使用__all__使得在外部使用from ... import *形式导入此模块时，这个模块里面只有add函数可以被外部发现
__all__ = ['add']

# 上面的__all__ 变量写在__init__.py文件里面表示用来对这个包下的module进行外部访问限制

def add(a, b):
    print(a + b)

def mult(a, b):
    print(a * b)


