# Filename: test3_recursion.py


def factorial(n):
    return 1 if n < 2 else n * factorial(n - 1)

# 尾递归优化
'''
尾递归是指，在函数返回的时候，调用自身本身，并且，return语句不能包含表达式。这样，编译器或者解释器就可以把尾递归做优化，使递归本身无论调用多少次，都只占用一个栈帧，不会出现栈溢出的情况。
'''
def fact(n):
    return fact_iter(n, 1)

def fact_iter(num, product):
    if num == 1:
        return product
    return fact_iter(num - 1, num * product)

def hanoi(n, source, auxiliary, target):
    '''
        汉诺塔，把a盘中的所有盘子通过b作为辅助盘全部移到c盘中
    :param n:  a中盘子的数量
    :param a:  a盘
    :param b:  b盘
    :param c:  c盘
    '''
    if n > 0:
        hanoi(n - 1, source, target, auxiliary)
        print(f'{source} -> {target}')
        hanoi(n - 1, auxiliary, source, target)


if __name__ == '__main__':
    print(factorial(5))
    # 递归栈溢出
    # RecursionError: maximum recursion depth exceeded in comparison
    # print(factorial(99999))
    # print(fact(1000))

    hanoi(3, 'A', 'B', 'C')
    # # A --> C
    # # A --> B
    # # C --> B
    # # A --> C
    # # B --> A
    # # B --> C
    # # A --> C