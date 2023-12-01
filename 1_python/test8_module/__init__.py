#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/1 14:50
# File      : __init__.py.py
# Project   : 1_python
# explain   :

'''
在Python中，一个.py文件就称之为一个模块（Module）

使用模块还可以避免函数名和变量名冲突。相同名字的函数和变量完全可以分别存在不同的模块中，因此，我们自己在编写模块时，不必考虑名字会与其他模块冲突。
但是也要注意，尽量不要与内置函数名字冲突。点这里[https://docs.python.org/3/library/functions.html]查看Python的所有内置函数。

你也许还想到，如果不同的人编写的模块名相同怎么办？为了避免模块名冲突，Python又引入了按目录来组织模块的方法，称为包（Package）

'''

''' 使用内置的sys模块来解析命令行参数 '''
import sys


def parse_args():

    ''' 解析命令行参数 '''
    # 从python 后面的后世命令参数
    args = sys.argv  # python .\test8_module\__init__.py --help ... --java c -format dot
    print(args)  # ['.\\test8_module\\__init__.py', '--help', '...', '--java', 'c', '-format', 'dot']
    args2 = sys.argv[1:] # ['--help', '...', '--java', 'c', '-format', 'dot']
    print(args2)


    pass

# 解析命令行，参数

def parse_comm_args():
    import argparse
    # 创建解析器对象
    parser = argparse.ArgumentParser(description='描述你的程序')
    # 添加命令行参数
    parser.add_argument('filename', help='输入文件名')
    parser.add_argument('-o', '--output', help='输出文件名')
    parser.add_argument('-t', '--type', help='类型')
    # 解析命令行参数
    args = parser.parse_args()
    # 访问解析后的参数
    print('输入文件名:', args.filename)
    print('输出文件名:', args.output)
    print('类型:', args.type)


if __name__ == '__main__':
    parse_args()


import socket

def get_local_ip():
    try:
        # 创建一个套接字对象
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # 连接一个外部服务器（可以是任何IP地址）
        local_ip = s.getsockname()[0]  # 获取本地IP地址
        s.close()  # 关闭套接字连接
        return local_ip
    except socket.error as e:
        print(f"Error occurred: {e}")
        return None

# 调用函数获取本机IP地址
ip_address = get_local_ip()
if ip_address:
    print(f"本机IP地址为: {ip_address}")
else:
    print("无法获取本机IP地址")



import requests

def get_public_ip():
    try:
        response = requests.get('https://api.ipify.org?format=json')  # 使用ipify服务获取公网IP
        if response.status_code == 200:
            public_ip = response.json()['ip']
            return public_ip
        else:
            print(f"Failed to fetch public IP. Status code: {response.status_code}")
            return None
    except requests.RequestException as e:
        print(f"Error occurred: {e}")
        return None

# 获取公网IP地址
public_ip_address = get_public_ip()
if public_ip_address:
    print(f"公网IP地址为: {public_ip_address}")
else:
    print("无法获取公网IP地址")


