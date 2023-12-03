#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/2 18:14
# Module    : test5.py
# explain   : 动态的给对象绑定属性和方法
import types


class Student:
    def __init__(self, name='None', age=18):
        # 下面的name，age是实列属性
        self.name = name
        self.age = age

    def to_string(self):
        attributes = ', '.join(f"{key} = {value}" for key, value in vars(self).items())
        return f"{self.__class__.__name__} = {{{attributes}}}"

stu1 = Student('张三', 18)
stu2 = Student('里斯', 28)

# 给某个实列对象动态绑定一个属性 addr
stu1.addr = '你家地址'
# 测试两个对象是否都有
print(stu1.to_string()) # Student = {name = 张三, age = 18, addr = 你家地址}
print(stu2.to_string()) # Student = {name = 里斯, age = 28}
# 说明addr属性动态的添加到了stu1身上


# 给某个实列对象动态绑定一个方法 run
def run(self):
    print(f'{self.name}在跑腿')
stu1.run = run

# 测试方法是否绑定上去了
stu1.run = types.MethodType(run, stu1) # 绑定
stu1.run() # 张三在跑腿
# stu2.run() # AttributeError: 'Student' object has no attribute 'run'


# 给某个类动态添加一个属性
setattr(Student, 'score', None)
stu1.score = 66
stu2.score = 77
print(stu1.to_string())
print(stu2.to_string())


# 给某个类动态添加一个方法
def eat(self):
    print(f'{self.name}在吃')
# 绑定方式1
Student.eat = eat
# 绑定方式2
# Student.eat = types.MethodType(eat, Student)
stu1.eat()
stu2.eat()


