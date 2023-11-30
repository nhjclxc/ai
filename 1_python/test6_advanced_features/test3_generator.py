# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/11/30 10:19
# File      : test3_generator.py
# Project   : 1_python
# explain   : 所以，如果列表元素可以按照某种算法推算出来，那我们是否可以在循环的过程中不断推算出后续的元素呢？这样就不必创建完整的list，
#   从而节省大量的空间。在Python中，这种一边循环一边计算的机制，称为生成器：generator。
# generator保存的是算法

''' 注意： 列表生成式使用的是[]，而生成器使用的则是() '''
from collections.abc import Iterable


def test_generator():
    print([x * x for x in range(10) if x % 2 == 0])  # [0, 4, 16, 36, 64]
    print((x * x for x in range(10) if x % 2 == 0))  # <generator object <genexpr> at 0x000002BF6B233680>

    g = (x * x for x in range(10) if x % 2 == 0)
    print(g.__next__())
    print(g.__next__())
    print(next(g))
    print(next(g))
    print(g.__next__())
    # print(g.__next__()) # StopIteration

    # 以下的for没有输出，因为g也是一个迭代器，而g的值在前面已经迭代输出完毕了，所以这里就没值，要重新生成过才会有值
    for v in g:
        print('for1 ', v)

    g = (x * x for x in range(10) if x % 2 == 0)
    for v in g:
        print('for2 ', v)

    pass

'''使用列表生成器生成：斐波拉契数列（Fibonacci）'''
# 除第一个和第二个数外，任意一个数都可由前两个数相加得到：  1, 1, 2, 3, 5, 8, 13, 21, 34, ...
def fib_test(max):
    n, a, b = 0, 0, 1
    while n < max:
        print(b)
        a, b = b, a + b
        n += 1

'''
这就是定义generator的另一种方法。如果一个函数定义中包含yield关键字，那么这个函数就不再是一个普通函数，而是一个generator函数，调用一个generator函数将返回一个generator：
最难理解的就是generator函数和普通函数的执行流程不一样。普通函数是顺序执行，遇到return语句或者最后一行函数语句就返回。而变成generator的函数，在每次调用next()的时候执行，遇到yield语句返回，再次执行时从上次返回的yield语句处继续执行。
'''
def fib_test_g(max):
    n, a, b = 0, 0, 1
    while n < max:
        # print(b)
        # 将print改成yield
        yield b
        a, b = b, a + b
        n += 1

# 调用generator函数会创建一个generator对象，多次调用generator函数会创建多个相互独立的generator。


def triangles(deep):
    '''
    杨辉三角定义如下：
              1
             / \
            1   1
           / \ / \
          1   2   1
         / \ / \ / \
        1   3   3   1
       / \ / \ / \ / \
      1   4   6   4   1
     / \ / \ / \ / \ / \
    1   5   10  10  5   1
    把每一行看做一个list，试写一个generator，不断输出下一行的list：
    '''

    x = 1
    super_l = []
    while x <= deep:
        curr_l = []
        if x == 1:
            curr_l = [1, ]
        else:
            i = 0
            while i < len(super_l) + 1:
                left = 0 if i == 0 else super_l[i-1]
                # if i == 0:
                #     left = 0
                # else:
                #     left = super_l[i-1]

                right = 0 if i == len(super_l) else super_l[i]
                # if i == len(super_l):
                #     right = 0
                # else:
                #     right = super_l[i]

                curr_l.append(left + right)
                i += 1
        x += 1
        super_l = curr_l
        print(curr_l)


def triangles_g(deep):
    x = 1
    super_l = []
    while x <= deep:
        curr_l = []
        if x == 1:
            curr_l = [1, ]
        else:
            i = 0
            while i < len(super_l) + 1:
                left = 0 if i == 0 else super_l[i-1]
                right = 0 if i == len(super_l) else super_l[i]

                curr_l.append(left + right)
                i += 1
        x += 1
        super_l = curr_l
        # print(curr_l)
        yield curr_l
def triangles_g2():
    r=[1]
    while True:
        yield(r)
        r = [1]+[r[i]+r[i+1] for i in range(len(r)-1)]+[1]

if __name__ == '__main__':
    # test_generator()

    # fib_test(7)
    # g = fib_test_g(7)
    # for v in g:
    #     print(v)

    # triangles(6)

    # var = triangles_g(6)
    # for v in var:
    #     print('v = ', v)

    gg = triangles_g2()

    print(isinstance(gg.__next__(), Iterable))

    n = 0
    results = []
    for t in triangles_g(10):
        results.append(t)
        n = n + 1
        if n == 10:
            break

    for t in results:
        print(t)

    if results == [
        [1],
        [1, 1],
        [1, 2, 1],
        [1, 3, 3, 1],
        [1, 4, 6, 4, 1],
        [1, 5, 10, 10, 5, 1],
        [1, 6, 15, 20, 15, 6, 1],
        [1, 7, 21, 35, 35, 21, 7, 1],
        [1, 8, 28, 56, 70, 56, 28, 8, 1],
        [1, 9, 36, 84, 126, 126, 84, 36, 9, 1]
    ]:
        print('测试通过!')
    else:
        print('测试失败!')


    pass


