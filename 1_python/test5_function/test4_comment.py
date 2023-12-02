#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/2 14:14
# File      : test4_comment.py
# Project   : 1_python
# explain   : 注释相关


# 1.函数注释
def add(a, b):
    """
        对两个数进行加法运算
    :param a: 参数1
    :param b: 参数2
    :return: 运算结果
    """
    return a + b


print(add(1, 5))
# 输出函数注释文档信息
help(add)
print(add.__doc__)

print(add(1.6, 2.5))

# 2.函数参数类型注释
# 分别给a，b和返回值加上了对应的类型注释，规定调用这个函数的参数的类型，但是不是限制死类型，你要传其他的也行
def add2(a: int, b: int = 6) -> int:
    """
        对两个数进行加法运算
    :param a: 参数1
    :param b: 参数2
    :return: 运算结构
    """
    return a + b

print(add2(2, 6))
print(add2(1.633, 2.5))

# 输出函数的类型注释
print(add2.__annotations__)

