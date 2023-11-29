

# 解析命令行，参数

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


