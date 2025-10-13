import os
from collections.abc import Iterable


def dict_iter():
    dict = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}

    for k in dict:
        print(k)

    for k in dict:
        print(k, dict.get(k))

    # dict默认是迭代key，如果要迭代value，则要使用dict.values()
    for v in dict.values():
        print(v)

    # key和value同时迭代，则需要使用dict.items()
    for k, v in dict.items():
        print(k, v)

    #

    pass


'''
for ... in ...循环要提供一个可迭代对象
'''


def test1():
    '''判断一个对象是不是可以迭代: from collections.abc import Iterable '''

    print(isinstance('qaz', Iterable))
    print(isinstance(('qaz', 123), Iterable))
    print(isinstance([], Iterable))
    print(isinstance({}, Iterable))
    print(isinstance({'k', 'v'}, Iterable))
    print(isinstance(123, Iterable))

    pass


def list_iter():
    '''
    如果要对list实现类似Java那样的下标循环怎么办？Python内置的enumerate函数可以把一个list变成索引-元素对，这样就可以在for循环中同时迭代索引和元素本身
    '''
    list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for i, v in enumerate(list):
        print(i, v)

    list2 = [('x1', 'y1'), ('x2', 'y2'), ('x3', 'y3')]
    for x, y in list2:
        print(x, y)

    list3 = [('x1', 'y1', 'z1'), ('x2', 'y2', 'z2'), ('x3', 'y3', 'z3')]
    for x, y, z in list3:
        print(x, y, z)

    pass


def findMinAndMax(L):
    '''
    请使用迭代查找一个list中最小和最大值，并返回一个tuple：
    :param L:
    :return:
    '''
    if not isinstance(L, Iterable):
        raise TypeError('对象不可迭代')
    if not L:
        return None, None

    min = max = L[0]
    for i in L:
        if i > max:
            max = i
        if i < min:
            min = i

    return (min, max)


def findMinAndMax_test():
    print(findMinAndMax([3.14, 0.168, 2.71828]))
    list = [9991, 2, 3, 4, 5, 6 - 9, 7, 8, 9]
    print(findMinAndMax(list))
    # 测试
    if findMinAndMax([]) != (None, None):
        print('测试失败!')
    elif findMinAndMax([7]) != (7, 7):
        print('测试失败!')
    elif findMinAndMax([7, 1]) != (1, 7):
        print('测试失败!')
    elif findMinAndMax([7, 1, 3, 9, 5]) != (1, 9):
        print('测试失败!')
    else:
        print('测试成功!')


def range_test():
    # 生成一个0，10的列表
    print(list(range(0, 10)))
    print(list(range(10)))

    # 生成一个5到10的列表
    print(list(range(5, 10)))

    # 步长为2
    print(list(range(10, 30, 2)))
    print(tuple(range(10, 30, 2)))

    ''' 如何生成 x*x的列表
        x * x for x in range(10)的意思就是说
            for后面的是遍历range的结果，而前面的x*x表示把后面遍历的值x拿过来计算得到最终的结果
        后面是列表生成式(数据生成)，而前面是结果计算式（遍历数据/重构数据）
    '''
    print(list(1 for x in range(10)))
    print(list(x for x in range(10)))
    print(list(x * x for x in range(10)))

    # 生成'ABC'和'xyz'的全排列
    print(list(m + n for m in 'ABC' for n in 'xyz'))  # 二层循环

    # 列出当前目录下的所有文件和目录名
    currIdr = os.listdir('../')
    print(list(d for d in currIdr))

    # 在列表生成式里面添加条件来限制生成的元素数据
    # 下面的意思是说：再生成数据的时候只取if成立的x来返回作为生成表达式的值
    print(list(x * x for x in range(10) if x % 2 == 0))

    # for后面的if不能带else，而for前面的表达式有if的话必须带上else
    # print(list(x * x for x in range(10) if x % 2 == 0 else 1))
    # 下面的意思是x%2==0成立的话输出的结果是x*x，否则输出1
    print(' if ... else ... for', list(x * x if x % 2 == 0 else 1 for x in range(10)))  # 先执行for range之后才执行if的三元表达式

    # 综上：可见，在一个列表生成式中，for前面的if ... else是表达式，而for后面的if是过滤条件，不能带else

    dict1 = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}
    for k, v in dict1.items():
        print(k, v)
    print(list(k + '=' + v for k, v in dict1.items()))

    pass


def test():
    L1 = ['Hello', 'World', 18, 'Apple', None]
    # L2 = list(x.lower() if isinstance(x, str) else x for x in L1)
    L2 = list(x.lower() for x in L1 if isinstance(x, str))
    print(L2)
    if L2 == ['hello', 'world', 'apple']:
        print('测试通过!')
    else:
        print('测试失败!')
    pass


if __name__ == '__main__':
    # dict_iter()
    #
    # test1()
    #
    list_iter()

    # findMinAndMax_test()

    # range_test()
    # test()

    # list、dict、str等数据类型不是Iterator
    # 集合数据类型如list、dict、str等是Iterable但不是Iterator，不过可以通过iter()函数获得一个Iterator对象。
    pass

    print("=============")

    dict1 = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}
    for k in dict1:
        print(k, dict1.get(k), dict1[k])
    print("=============")
    for k, v in dict1.items():
        print(k, v)
