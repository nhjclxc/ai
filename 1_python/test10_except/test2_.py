#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/2 14:42
# File      : test2_.py
# Project   : 1_python
# explain   : 捕获异常

'''
try:
    可能发生异常的语句
except: 异常类型1 as 类型别名:
    对异常类型1的处理
except: (异常类型2, 异常类型3) as 类型别名: # 使用元组同时捕获异常类型2, 异常类型3，对这两个类型执行相同的操作
    对异常类型2、3的处理
except: 异常类型4 as 类型别名:
    对异常类型4的处理
else:
    try里面没有发生异常时要执行的代码
finally:
    无论是否发生异常都会执行的语句

注意：异常类型1，2，3，4是从小到大的


在 Python 中，异常对象通常具有一些常用的属性，这些属性可以提供有关异常的信息。以下是一些常用的异常对象属性：
    args：包含异常相关的参数，可能是一个字符串，一个异常实例，或者一个包含有关异常的元组。
    __str__() 或 __repr__()：这些方法用于返回异常的字符串表示，通常包含异常的详细信息。
    __cause__：指向引发当前异常的原因异常，如果存在的话。
    __context__：指向导致当前异常的上下文异常，如果存在的话。
    with_traceback()：设置异常的回溯信息（traceback）。
    __class__：指向异常的类。
    __doc__：异常的描述文档字符串，提供了异常的解释和信息。
    __module__：异常所在的模块名称。

输出异常堆栈信息
    首先：import traceback
    其次输出：traceback.print_tb(e.__traceback__)或使用traceback.print_exc()
'''
import traceback

fp = None
try:
    # fp = open('hello.txt', 'w', encoding='UTF-8')
    fp = open('hello.txt', 'r', encoding='UTF-8')
    print(111)
except FileNotFoundError as e:
    print('出现文件不存在异常')
except Exception as e:
    print('出现了异常')
    print(e.args)
    traceback.print_tb(e.__traceback__)
    print()
    traceback.print_exc()
else:
    print('没有发生异常')
finally:
    if fp:
        fp.close()