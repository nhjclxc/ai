import math


def fac(n):
    if n == 0 | n == 1:
        return 1
    else:
        return fac(n - 1) * n

def factorial(n):
    return 1 if n < 2 else n * factorial(n - 1)


def factorial2(n, accumulator=1):
    if n < 0:
        return None  # 对负数返回空值或者抛出异常，视情况而定
    if n in (0, 1):
        return accumulator
    return factorial2(n - 1, accumulator * n)


def factorial3(n):
    if n < 0:
        return None  # 对负数返回空值或者抛出异常，视情况而定
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

def my_abs(x):
    if x > 0:
        return x
    else:
        return -x
def my_abs2(x):
    return x if x > 0 else -x

def quadratic(a, b, c):
    '''
        用求根公式解一元二次方程
    :param a:
    :param b:
    :param c:
    :return:
    '''
    if not isinstance(a, (int, float)) \
            or not isinstance(b, (int, float)) \
            or not isinstance(c, (int, float)) :
        raise RuntimeError('数据类型不匹配')
    detar = b * b - 4 * a * c
    if detar < 0:
        # raise RuntimeError('此方程无解')
        sq = math.sqrt(-detar)
        return complex(-b/2*a, sq), complex(-b/2*a, -sq)

    sq = math.sqrt(detar)
    x1 = (-b + sq) / (2 * a)
    x2 = (-b - sq) / (2 * a)
    return (x1, x2)

default_n = 2

def power(x, n=default_n):
    s = 1
    while n > 0:
        s *= x
        n -= 1
    return s

# 定义默认参数要牢记一点：默认参数必须指向不变对象！
''' 默认参数必须指向不可变对象 '''
def test_default_param(list = []):
    ''' 测试可变参数作为默认参数 '''
    list.append('end')
    return list

def test_default_param_2(list=None):
    ''' 测试可变参数作为默认参数 '''
    if list is None:
        list = []
    list.append('end')
    return list


if __name__ == '__main__':
    # print('quadratic(2, 3, 1) =', quadratic(2, 3, 1))
    # print('quadratic(1, 3, -4) =', quadratic(1, 3, -4))
    #
    # if quadratic(2, 3, 1) != (-0.5, -1.0):
    #     print('测试失败1')
    # if quadratic(1, 3, -4) != (1.0, -4.0):
    #     print('测试失败2')
    #
    # print('\n\n\n')
    # print(fac(5))
    #
    # # 函数赋值，我在使用的时候认为某一个函数的名字没取号，但是又不想修改原来的函数名（可能别人在用）
    # # 这时我就可以在自己的代码空间内修改该函数名
    # # 这时我就可以在自己的代码空间内修改该函数名
    # my_fac = factorial
    # print(my_fac(5))
    #
    # print(hex(0))
    # print(hex(1))
    # print(hex(512))
    #
    # abs(55)
    #
    #
    # print(my_abs(-9))
    # print(my_abs2(-9))

    # print(power(2))
    # print(power(3, 2))
    # print(power(3, 3))
    # print(power(3, -1))

    print(test_default_param()) # ['end']
    print(test_default_param()) # ['end', 'end']
    print(test_default_param()) # ['end', 'end', 'end']
    print(test_default_param_2())
    print(test_default_param_2())
    print(test_default_param_2())






