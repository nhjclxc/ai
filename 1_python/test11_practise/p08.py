import random
import time
from functools import wraps


def download(filename):
    """下载文件"""
    print(f'开始下载{filename}.')
    time.sleep(random.random() * 3)
    print(f'{filename}下载完成.')


def upload(filename):
    """上传文件"""
    print(f'开始上传{filename}.')
    time.sleep(random.random() * 2)
    print(f'{filename}上传完成.')


def test1():
    # 不使用装饰器的情况下

    # 直接调用
    # download('MySQL从删库到跑路.avi')
    # upload('Python从入门到住院.pdf')

    # 新需求：现在要统计上面两个函数个花了多少时间
    t1 = time.time()
    download('MySQL从删库到跑路.avi')
    t2 = time.time()
    upload('Python从入门到住院.pdf')
    t3 = time.time()
    print("download 花费了 ", t2 - t1)
    print("upload 花费了 ", t3 - t2)


# 使用装饰器优化
def wrapper_fun(func):
    # 如果在代码的某些地方，我们想去掉装饰器的作用执行原函数，那么在定义装饰器函数的时候，需要做一点点额外的工作。
    # Python 标准库functools模块的wraps函数也是一个装饰器，我们将它放在wrapper函数上，这个装饰器可以帮我们保留被装饰之前的函数，
    # 这样在需要取消装饰器时，可以通过被装饰函数的__wrapped__属性获得被装饰之前的函数。

    @wraps(func)
    def wrapper(*args, **kwargs):
        # 目标函数执行前的操作
        st = time.time()
        # 执行被包装的函数
        res = func(*args, **kwargs)
        # 目标函数执行完毕之后的操作
        et = time.time()
        print(f"函数{func}, 耗时 {et - st}")
        # 返回目标函数的返回值（如果目标函数原本就有的花）
        return res
    return wrapper


# 使用定义好的装饰器来包裹函数
@wrapper_fun
def download2(filename):
    """下载文件"""
    print(f'开始下载{filename}.')
    time.sleep(random.random() * 3)
    print(f'{filename}下载完成.')


def upload2(filename):
    """上传文件"""
    print(f'开始上传{filename}.')
    time.sleep(random.random() * 2)
    print(f'{filename}上传完成.')


def test2():

    download2('MySQL从删库到跑路.avi')
    upload2('Python从入门到住院.pdf')

    # 取消装饰器的作用不记录执行时间
    download2.__wrapped__('MySQL必知必会.pdf')

    print('-----包裹函数------')
    w_download = wrapper_fun(download2)
    w_upload = wrapper_fun(upload2)

    w_download('MySQL从删库到跑路.avi')
    w_upload('Python从入门到住院.pdf')

    pass


    # 函数作为装饰器
if __name__ == '__main__':
    # test1()

    test2()

