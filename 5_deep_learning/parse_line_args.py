#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/27 18:26
# Module    : parse_line_args.py
# explain   : 解析命令行参数 解析 命令行 参数 命令行参数


import argparse

def fun(data):
    print('fun ', data)
    pass

if __name__ == '__main__':
    # python parse_line_args.py --input input_file.txt --output output_file.txt --verbose
    # 创建解析器
    parser = argparse.ArgumentParser(description='Argument Parser Example')

    # 添加命令行参数
    parser.add_argument('--input', type=str, default= 'temp_dot_file.dot' ,help='Input file path')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose mode')

    # 解析命令行参数
    args = parser.parse_args()

    # 使用命令行参数
    if args.input:
        print('Input file path:', args.input)
        fun(args.input)

    if args.output:
        print('Output file path:', args.output)

    if args.verbose:
        print('Verbose mode enabled')
