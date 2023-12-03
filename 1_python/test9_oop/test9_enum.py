#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/3 15:21
# Module    : test9_enum.py
# explain   : 定义py的枚举类型

'''
这样的枚举类型定义一个class类型，然后，每个常量都是class的一个唯一实例。Python提供了Enum类来实现这个功能：

'''

from enum import Enum, unique

color = Enum('Color', ('red', 'green', 'blue'))

print(color)
print(color.red)
for name, item in color.__members__.items():
    print(name, item )
#     value属性则是自动赋给成员的int常量，默认从1开始计数


print()
print()
# 自定义枚举类型
# @unique装饰器可以帮助我们检测枚举类型是否有重复的值，在每一个枚举类上最好都要加上这个装饰器
@unique
class ColorEnum(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3
    # BLACK = 3 # ValueError: duplicate values found in <enum 'ColorEnum'>: BLACK -> BLUE

print(ColorEnum.__members__)

for name, item in ColorEnum.__members__.items():
    print(name, item,  item.name, item.value)



print(ColorEnum.RED)
print(ColorEnum.GREEN)
print(ColorEnum.BLUE)

red = ColorEnum.RED
print(red == ColorEnum.RED)
print(red is ColorEnum.RED)
print(red)
print(ColorEnum.RED)
print(type(red))
print(type(ColorEnum.RED))
print(id(red))
print(id(ColorEnum.RED))



@unique
class Gender(Enum):
    Male = 0
    Female = 1

class Student(object):
    def __init__(self, name, gender):
        self.name = name
        self.gender = gender
# 测试:
bart = Student('Bart', Gender.Male)
if bart.gender == Gender.Male:
    print('测试通过!')
else:
    print('测试失败!')