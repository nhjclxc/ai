#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/1 16:18
# File      : hello.py
# Project   : 1_python
# explain   : 作为模块导入的被测试文件


print(__name__)

# 直接启动这个文件输出：__main__
# 当被作为模块导入其他文件时输出：hello（就是这个文件的文件名，也就是这个模块的命名空间）

# 根据上面的输出，进而说明了在本文件自己启动的时候就是固定的"__main__"，而在import后的__name__是文件名
