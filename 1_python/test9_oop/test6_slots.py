#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/2 18:14
# Module    : test6_slots.py
# explain   : 使用__slots__来控制对象可以设置哪些属性



class Student:
    # 限制实列对象只能动态的添加下列属性，不能无限制的添加下去
    # __slots__定义的属性仅对当前类实例起作用，对继承的子类是不起作用的：
    # 当父类子类都有__slots__的声明时，子类的属性时父类子类的并集
    __slots__ = ('name', 'age', 'addr')
    def __init__(self, name='None', age=18):
        # 下面的name，age是实列属性
        self.name = name
        self.age = age


# 参数在__slots__上面添加一个没有声明 的属性
stu1 = Student('张三', 18)
stu1.addr = '地球'
print(stu1.addr)
stu1.addr22 = '地球22' #AttributeError: 'Student' object has no attribute 'addr22'
print(stu1.addr22)

