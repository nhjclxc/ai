#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/2 17:40
# Module    : test_test1.py
# explain   :

# import test1
# test1.add(2, 6)
# test1.mult(2,6)

from test1 import * # *通过test1里面的__all__来定义，默认时全部内容
add(2, 6)
# mult(2,6) #NameError: name 'mult' is not defined


