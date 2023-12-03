#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/3 11:27
# Module    : test7_property.py
# explain   : @property装饰器的使用


# 既能检查参数，又可以用类似属性这样简单的方式来访问类的变量
class Rectangle(object):

    def __init__(self, weigth, heigth):
        self.weigth = weigth
        self.heigth = heigth

    '''
    很多时候，有一些属性是不希望外界可以通过“实列变量.属性名”，这种方式来修改的
        因此，这个时候就要限制外界随意的修改属性，可以使用@property装饰器

    @property作用：
        当某个方法被@property修饰之后，这个方法其实是被当成一个属性对外部来使用，外部使用的时候就是通过“实列变量.函数名”实现了“实列变量.属性名”的功能
        这时候这个属性是一个只读的状态。那么如何实现可以修改呢？
        实现修改的方法：
            @'只读属性的函数名'.setter，这个'只读属性的函数名'表示的是被@property修饰的函数，通过这个setter装饰器来为被@property修饰的函数属性添加一个set方法，进而达到修改该函数属性的目的
    '''

    @property
    def area(self):
        return self.weigth * self.heigth


rec = Rectangle(2, 6)
print(rec.weigth)
print(rec.heigth)
print(rec.area)

# 实现修改的方法：
class Book(object):
    '''
        使用price来对外暴露
        使用_price来在内部存储
    '''
    def __init__(self, name: str, price: float):
        self.name = name
        self._price = price

    # 注意被封装的属性名不要和封装的函数名冲突，否则会发生无限递归调用
    # 使用_属性名来表示内部的变量，去除下划线表示允许外部访问。'_属性名'形式定义的属性，在py中约定俗成不能取修改
    # 装饰price
    @property
    def price(self):
        return self._price

    # 为封装过后的price添加一个修改方法
    @price.setter
    def price(self, price):
        if not isinstance(price, float):
            raise RuntimeError('类型错误')
        if price < 0:
            raise RuntimeError('价格必须大于0')
        self._price = price

'''
通过使用@property和@xxx.setter来对某个属性进行封装之后，外部外部无法直接修改被封装的属性
'''

# 测试
book1 = Book('一篇小说', 16.36)
print(book1.name)
print(book1.price) # 直接像使用正常属性一样
# print(book1._price)
book1.price = 52.6 # 直接像修改属性一样修改
print(book1.price)
# book1.price = '52.6' # RuntimeError: 类型错误







# 请利用@property给一个Screen对象加上width和height属性，以及一个只读属性resolution：
class Screen(object):
    def __init__(self):
        pass

    @property
    def width(self):
        return self.__width
    @width.setter
    def width(self, w):
        self.__width = w

    @property
    def height(self):
        return self.__height
    @height.setter
    def height(self, h):
        self.__height = h

    @property
    def resolution(self):
        return self.width * self.height
    pass
# 测试:
s = Screen()
s.width = 1024
s.height = 768
print('resolution =', s.resolution)
if s.resolution == 786432:
    print('测试通过!')
else:
    print('测试失败!')



