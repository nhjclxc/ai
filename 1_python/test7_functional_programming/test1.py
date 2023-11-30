# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/11/30 14:34
# File      : test1.py
# Project   : 1_python
# explain   :

# 变量可以是函数，函数也可以是变量

'''
# 由于变量可以是函数，函数有可以是变量
#   那么，一个函数的参数也可以是某一个函数，那么这种在参数上面带有参数函数的函数就叫做高阶函数
'''


def add(x, y, fun):
    # 当传入为abs时，以下相当于：abs(x) + abs(y)
    return fun(x) + fun(y)


def square(x):
    return x ** 2


def add2(x, y):
    # 当传入为abs时，以下相当于：abs(x) + abs(y)
    return abs(x) + abs(y)


def test_map():
    # 现在想实现一个功能，返回list = [1,2,3,4,5,6,7,8,9]对应的平方值
    lst = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    list2 = [x * x for x in lst]
    print(list2)

    # 现在要求通过函数式编程Lambda来实现 , map实现
    # map函数的作用，会对列表里面的每一个值进行fun对应的操作

    """
    map(func, *iterables) --> map object
        Make an iterator that computes the function using arguments from
        each of the iterables.  Stops when the shortest iterable is exhausted.
    """
    iter = map(square, lst)
    print(type(iter))  # 可迭代对象
    print(list(iter))
    for s in iter:
        print(s)

    # *iterables传入了多少个值，那么对应的函数func就要可以接收多少个参数，fun会按顺序的去对应的可迭代对象里面拿数据
    # 并且当两个可迭代对象的长度不同时，取长度短的某一个可迭代对象为计算映射的个数
    print(lst)
    print(list2)
    iter2 = map(add2, lst, list2)
    print(type(iter2))  # 可迭代对象
    print(list(iter2))

    pass


def fun1(x, y):
    return x * 10 + y


from functools import reduce


# reduce可以作为求一个序列的和
def test_reduce():
    # 要求，reduce函数接收的函数参数必须要可以接收两个参数
    # 对传入的两个参数进行操作

    lst = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    # reduce(function, iterable[, initial]) -> value
    print(reduce(add2, lst))

    print(reduce(fun1, lst))

    pass


def is_even(x):
    '''判断x是不是偶数'''
    # return True if x % 2 == 0 else False
    return x % 2 == 0


def is_empty(s):
    return s and s.strip()


def test_filter():
    """
    filter(function or None, iterable) --> filter object

    Return an iterator yielding those items of iterable for which function(item)
    is true. If function is None, return the items that are true.

    使用filter函数的参数参数只能接收一个参数，当函数参数返回True时保留对应的列表值，返回False时去除该值
    """

    # lst = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # print(list(filter(is_even, lst)))

    # 实现去除一个列表里面的空字符串

    # 空None、空字符串''和全部是空格的字符串'        '.strip()，使用if时都是True
    if not '':
        print('True1')
    if not '        '.strip():
        print('True2')

    lst2 = ['  ', ' sa ', 'sc  ', None, '  s', '']
    print(list(filter(is_empty, lst2)))

    pass


def test_sorted():
    """
    Return a new list containing all items from the iterable in ascending order.

    A custom key function can be supplied to customize the sort order, and the
    reverse flag can be set to request the result in descending order.
    """
    lst = [1, -2, 3, 626, 9, 4, 5, 6, -7, 8, 9]
    print(sorted(lst))  # 默认升序
    print(sorted(lst, reverse=True))  # 降序

    s_lsit = ['C', 'a', 'z', 'd', 'A', 'w']
    print(sorted(s_lsit))  # 按字符ASCII升序

    # 自定义排序，要求按绝对值大小进行排序
    lst2 = [1, -1, 2, -2, 3, 9, -9, -5]
    print(sorted(lst2, key=abs))

    pass


def test_lambda():
    '''
        lambda格式：
            lambda arg1, arg2, arg3 ... : statement
        注意：前面的'lambda'一定要写
    '''

    # ff指向了某个匿名函数
    ff = lambda a, b, c: a + b + c
    ff2 = lambda a, b, c: (print('l 1'), print('l 2'), a + b + c)

    print(ff(1, 5, 9))
    print(ff2(1, 5, 9))

    pass


class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def to_string(self):
        print(f'name = {self.name}, age = {self.age}')


def test_class_sorted():
    """使用sorted对自定义对象进行排序"""
    s1 = Student('xhangsan ', 15)
    s2 = Student('lisi ', 11)
    s3 = Student('wangwu ', 19)

    sl = sorted([s1, s2, s3], key=lambda a: a.age)
    for s in sl:
        s.to_string()

    pass


'''
利用map()函数，把用户输入的不规范的英文名字，变为首字母大写，其他小写的规范名字。
输入：['adam', 'LISA', 'barT']，输出：['Adam', 'Lisa', 'Bart']：
'''


def normalize(name):
    return name[0].upper() + (name[1:].lower())


def prod(L):
    if not L:
        return None
    return reduce(lambda x, y: x * y, L)


# 利用map和reduce编写一个str2float函数，把字符串'123.456'转换成浮点数123.456：
def str2float(s):
    return map(lambda x: float(x), s)

''' https://zhuanlan.zhihu.com/p/453787908
在一些语言中，在函数中可以（嵌套）定义另一个函数时，如果内部的函数引用了外部的函数的变量，
则可能产生闭包。闭包可以用来在一个函数与一组“私有”变量之间创建关联关系。
在给定函数被多次调用的过程中，这些私有变量能够保持其持久性。
'''


# 利用闭包返回一个计数器函数，每次调用它返回递增整数：
def createCounter():
    x = 0
    def counter():
        nonlocal x
        x += 1
        return x
    return counter

if __name__ == '__main__':
    # 把某一个函数作为参数传入，那么这一条语句就是函数式编程
    # print(add(-9, 6, abs)) #将取绝对值的函数作为参数传入

    # test_map()

    # test_reduce()

    # test_filter()

    # test_sorted()

    # test_class_sorted()

    # test_lambda()

    # L1 = ['adam', 'LISA', 'barT']
    # L2 = list(map(normalize, L1))
    # print(L2)

    # print('3 * 5 * 7 * 9 =', prod([3, 5, 7, 9]))
    # if prod([3, 5, 7, 9]) == 945:
    #     print('测试成功!')
    # else:
    #     print('测试失败!')

    # print(type(str2float('12.365')))
    # print(str2float('12.365'))

    counterA = createCounter()
    print(counterA(), counterA(), counterA(), counterA(), counterA())  # 1 2 3 4 5
    counterB = createCounter()
    if [counterB(), counterB(), counterB(), counterB()] == [1, 2, 3, 4]:
        print('测试通过!')
    else:
        print('测试失败!')

    pass
