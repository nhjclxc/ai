#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/2 14:33
# File      : test1.py
# Project   : 1_python
# explain   : 了解异常

f = open('hello.txt', 'r', encoding='UTF-8')


'''
Traceback (most recent call last):
  File "E:\nbu\ai\1_python\test10_except\test1.py", line 9, in <module>
    f = open('hello.txt', 'r', encoding='UTF-8')
FileNotFoundError: [Errno 2] No such file or directory: 'hello.txt'
'''

# File "E:\nbu\ai\1_python\test10_except\test1.py" 表示出错的文件
# line 9, in <module> 表示出错的具体位置
# f = open('hello.txt', 'r', encoding='UTF-8') 表示出现异常的具体位置
# FileNotFoundError 表示异常类型
# 表示出错的异常编号
# No such file or directory: 'hello.txt' 表示出现异常的具体原因