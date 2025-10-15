import functools


def power(base, exp):
    return base ** exp


def test1():
    # 🧩 1️⃣ functools.partial —— 偏函数
    # 可以 固定函数的部分参数，生成一个新函数。

    p2 = functools.partial(power, exp=2)
    p3 = functools.partial(power, exp=3)

    print(p2(2))
    print(p3(2))

    pass


def test2():
    # 🧩 2️⃣ functools.lru_cache —— 缓存函数结果
    # 自动缓存函数调用结果，提高性能（特别是递归函数）。

    functools.lru_cache()

    pass


if __name__ == '__main__':
    """
    functools --- 函数操作工具箱
    Python 的 functools 模块是标准库中专门用于 函数操作和函数式编程 的工具箱，包含缓存、偏函数、装饰器等实用功能。
    
    """

    # test1()

    test2()
