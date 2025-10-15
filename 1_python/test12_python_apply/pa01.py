import os
import traceback
from datetime import time, datetime


def test1():
    """ 读文件 """
    """
    操作模式	具体含义
    'r'	读取 （默认）
    'w'	写入（会先截断之前的内容）
    'x'	写入，如果文件已经存在会产生异常
    'a'	追加，将内容写入到已有文件的末尾
    'b'	二进制模式
    't'	文本模式（默认）
    '+'	更新（既可以读又可以写）
    """

    file = open("file_opt.txt", "r", encoding="utf-8")
    print(file.readline())  # readline 每次读取一行文件里面的内容
    print(file.readline())
    print(file.readline())

    # 遍历文件的每一行内容
    # for line in file:
    #     print(line)

    # lines = file.readlines()
    # for line in lines:
    #     print(line)

    file.close()


    # 写文件
    file2 = open("test.log", "a", encoding="utf-8")
    file2.write("时间: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")


    pass


def test2():
    """ 文件读取的异常处理 """

    # 如果 file.txt 文件不存在就会出现以下问题，程序会奔溃
    # FileNotFoundError: [Errno 2] No such file or directory: 'file.txt'
    # file = open("file.txt", "r", encoding="utf-8")


    # 加入异常处理机制

    file = None

    try:
        file = open("file.txt", "r", encoding="utf-8")
    except FileNotFoundError as fnfe:
        print("文件未找到异常：" + fnfe.strerror + "")
        traceback.print_exc()  # # ✅ 打印完整堆栈，且程序正常往下执行
        tb = traceback.TracebackException.from_exception(fnfe)
        print("".join(tb.format()))
        err_log = open("err_log.log", "a", encoding="utf-8")
        err_log.write("".join(tb.format()))
        err_log.close()
    except RuntimeError as rte:
        print("运行时异常：" + rte.strerror + "")
    except Exception as e:
        print("未知异常：" + e.strerror + "")
    else:
        print("没有遇到任务异常")
    finally:
        if file is not None:
            file.close()

    print("运行结束")


def test3():
    # 上下文管理器语法

    try:
        # with会自动关闭文件，无须手动关闭文件，就不需要写finally代码块了
        with open("file_opt.txt", "r", encoding="utf") as file:
            print(file)
            size = os.path.getsize(file.name)
            print(f"文件大小：{size} 字节")
    except FileNotFoundError as fnfe:
        print(fnfe.strerror + "")

    pass


# ZeroDivisionError: division by zero
class MyZeroDivisionError(Exception):

    def __init__(self, msg="my除0异常"):
        super().__init__(msg)


def test5():
    # 自定义异常类型

    x = 1
    if x == 0:
        raise MyZeroDivisionError()
        # raise MyZeroDivisionError("分母不能为0")

    x = 1
    try:
        if x == 0:
            raise MyZeroDivisionError()
    except MyZeroDivisionError as e:
        print("补货到了自定义异常")
    else:
        print("无事发生")
    finally:
        print("最后 finally")


    pass


def test6():
    # 读写二进制数据
    try:
        data = None
        with open("test.log", "rb") as file:
            data = file.read()
        with open("test2.log", "wb") as file:
            file.write(data)
    except FileNotFoundError as fnfe:
        print(fnfe.strerror + "")


    pass


def copy_file(src, dest):
    """ 文件复制 """

    try:
        data = None
        # 读
        with open(src, 'rb') as file:
            data = file.read()
        # 写
        with open(dest, 'wb') as file:
            file.write(data)

        # 使用一个with同时打开两个文件
        with open(src, 'rb') as fsrc, open(dest, 'wb') as fdst:
            fdst.write(fsrc.read())
    except FileNotFoundError as fnfe:
        print(fnfe.strerror + "")

    print("文件复制成功！！！")


def copy_file2(src, dest, buffer_size=1024 * 1024):
    """文件复制（流式复制，节省内存）"""
    try:
        with open(src, 'rb') as fsrc, open(dest, 'wb') as fdst:
            while True:
                chunk = fsrc.read(buffer_size)
                if not chunk:
                    break
                fdst.write(chunk)
        print("✅ 文件复制成功！")
    except FileNotFoundError:
        print(f"❌ 文件未找到：{src}")
    except Exception as e:
        print(f"⚠️ 发生未知错误：{e}")


import shutil

# ✅ 版本 3：使用 shutil 标准库（最 Pythonic）
# Python 自带 shutil，专门为文件操作设计，非常简洁：
def copy_file3(src, dest):
    """ 文件复制（使用标准库）"""
    try:
        shutil.copyfile(src, dest)
        print("✅ 文件复制成功！")
    except FileNotFoundError:
        print(f"❌ 文件未找到：{src}")
    except Exception as e:
        print(f"⚠️ 发生未知错误：{e}")


def test8():


    copy_file("img.png", "img22.png")


    pass



if __name__ == '__main__':

    # test1()

    # test2()

    # test3()

    # test5()

    test6()

    test8()

    pass

