#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/1 19:38
# File      : test2.py
# Project   : 1_python
# explain   :


'''
如果要让内部属性不被外部访问，可以把属性的名称前加上两个下划线__，在Python中，实例的变量名如果以__开头，
就变成了一个私有变量（private），只有内部可以访问，外部不能访问，注意：这个只是约定俗成的不能访问，如果你非要访问的话，还是可以访问的

'''

class Test2():
    def __init__(self, __height, weight):
        self.__height = __height
        self.weight = weight

    ''' 使用双下划线的属性或是函数是私有变量，外部不能直接访问 '''

    def __inner_func(self):
        print('__inner_func')

    def get_height(self):
        return self.__height

test2 = Test2(169, 120)
# print(test2.__height) #AttributeError: 'Test2' object has no attribute '__height'
print(test2.weight)
# test2.__inner_func() #AttributeError: 'Test2' object has no attribute '__inner_func'


'''间接访问类的私有变量和函数'''
# Python 会将私有属性重命名为 _ClassName__attribute 的形式，其中 ClassName 是类的名称，attribute 是属性名称。这种方式允许你通过修改名称来访问私有属性。
# 因此可以通过 "_ClassName__attribute"的方式来访问私有变量
print(test2._Test2__height)
# 通过"_ClassName__inner_func(args)"来访问私有函数
test2._Test2__inner_func()
