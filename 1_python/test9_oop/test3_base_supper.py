#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/1 20:14
# File      : test3_base_supper.py
# Project   : 1_python
# explain   : 继承和多态


class Animal():
    def __init__(self):
        pass

    def run(self):
        print(f'{self.__class__.__name__} is runing')


animal = Animal()


class Dog(Animal):
    def __init__(self):
        super().__init__()


dog = Dog()
dog.run()
print(isinstance(dog, Dog))
print(isinstance(dog, Animal))


class Cat(Animal):
    def __init__(self):
        super().__init__()


cat = Cat()
cat.run()
print(isinstance(cat, Cat))
print(isinstance(cat, Animal))


# 测试多态
def test_animal_run(animal):
    animal.run()

test_animal_run(animal)
test_animal_run(dog)
test_animal_run(cat)



print()
print()
'''
py的多继承
'''

# py的所有class默认继承 object
class Class1(object):
    def __init__(self, name):
        print('Class1.__init__', name)
        super(Class1, self).__init__(name)
        self.name = name

    def run(self):
        print(f'{self.name}正在跑步')

class Class2(object):
    def __init__(self, name, age):
        print('Class2.__init__', name, age)
        super(Class2, self).__init__()
        self.name = name
        self.age = age

    def fly(self):
        print(f'{self.name}正在飞翔')

class Class3(Class1, Class2):
    '''
        第一个类被称为继承主类（直接父类），主要的属性来自第一个类，其余的父类用于提供一些功能，如接口，
            而功能的接口类名一般以MinIn结尾，如：RunnableMixIn、FlyableMixIn
    '''
    def __init__(self, name, age):
        print('Class3.__init__')
        # 下面是对每一个父类进行分开调用他们的init方法
        # super().__init__(name)  # 调用第一个父类的 __init__ 方法
        Class1.__init__(self, name)  # 调用第一个父类的 __init__ 方法
        Class2.__init__(self, name, age)  # 调用第二个父类的 __init__ 方法
        # 下面是一起调用Class3的父类方法，其中__init__(args)，里面的参数表示的就是奔雷即Class3的初始化方法的参数（注意：不是父类的）
        # super(Class3, self).__init__(name, age)

c3 = Class3('测试3',18)

class Base(object):
    def __init__(self):
        print ("enter Base")
        print ("leave Base")

class A():
    def __init__(self):
        print ("enter A")
        super(A, self).__init__()
        print ("leave A")

class B():
    def __init__(self):
        print ("enter B")
        super(B, self).__init__()
        print ("leave B")

class C(A, B):
    def __init__(self):
        print( "enter C")
        super(C, self).__init__()
        print ("leave C")

c = C()



