# Filename: func2_returns.py
from arrow import now


def fun1():
    ''' 测试函数的多个返回值 '''

    # 由于tuple元组()的不可变性，所以多个函数返回值一般是使用元组来封装返回
    return (111,'qwe',now())


if __name__ == '__main__':
    tuple = fun1()
    for t in tuple:
        print(t)



