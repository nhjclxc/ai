#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/3 14:43
# Module    : test8.py
# explain   : 定制类


class Student:
    def __init__(self, name='None', age=18):
        self.name = name
        self.age = age

    # def to_string(self):
    #     attributes = ', '.join(f"{key} = {value}" for key, value in vars(self).items())
    #     return f"{self.__class__.__name__} = {{{attributes}}}"

    # 在py中的to_string方法是：__str__
    def __str__(self):
        attributes = ', '.join(f"{key} = {value}" for key, value in vars(self).items())
        return f"{self.__class__.__name__} = {{{attributes}}}"


stu = Student('张三', 18)
# print(stu.to_string())

print(stu)
print(stu.__str__())





'''
如果一个类想被用于for ... in循环，类似list或tuple那样，就必须实现一个__iter__()方法，该方法返回一个迭代对象，
然后，Python的for循环就会不断调用该迭代对象的__next__()方法拿到循环的下一个值，直到遇到StopIteration错误时退出循环。
'''
# 如果一个类想被用于iterator遍历，则必须实现一个__iter__()
class Fib(object):
    def __init__(self, max):
        # 初始斐波那契数列的前两个变量
        self.a, self.b = 0, 1
        self.max = max
        self.counter = 0

    # 定义迭代器对象
    def __iter__(self):
        # 迭代器方法直接返回本对象就好了，后面在迭代的时候直接调用__next()__即可
        return self

    def __next__(self):
        self.counter += 1

        if self.counter == self.max + 1:
            raise StopIteration()
        elif self.counter == 1:
            return self.a
        elif self.counter == 2:
            return self.b

        temp = self.a + self.b
        self.a = self.b
        self.b = temp
        return temp

fib = Fib(8)
for f in fib.__iter__():
    print(f)

print()

# 如果想实现通过下标获取元素的值，那么就要实现__getitem__()
class Fib2(object):
    def __init__(self):
        # 初始斐波那契数列的前两个变量
        pass

    def __getitem__(self, index):
        '''
            在内部实现斐波那契数列的求职
        :param index: 索引
        :return: 索引对应的值
        '''
        # 下面的for循环表示对0到index的数值进行for遍历 ，类似于for(int i = 0; i < index; i++){}
        a, b = 0, 1
        if index == 0:
            return a
        elif index == 1:
            return b
        temp = None
        for i in range(index - 1):
            temp = a + b
            a = b
            b = temp
            # a, sb = self.b, a + b
        return temp

fib2 = Fib2()
print(fib2[0])
print(fib2[1])
print(fib2[2])
print(fib2[3])
print(fib2[4])
print(fib2[5])
print(fib2[6])
print(fib2[7])
print(fib2[8])

# __getattr__()，可以动态的返回一个对象的属性




