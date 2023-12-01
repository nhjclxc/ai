# Filename: test3_io.py  标准输入输出


'''
    标准输入：input
    输入数据的格式是字符串 str
'''

str = input('输入str：')
print(f'str = {str}, type(str) = {type(str)}，id(str) = {id(str)}')
num = int(input('输入num：'))
print(f'str = {num}, type(str) = {type(num)}，id(str) = {id(num)}')


'''
    标准输出：print
        print(self, *args, sep=' ', end='\n', file=None): 
'''

# 打开文件以写入模式
with open('../output.txt', 'w') as file:
    print('Hello, world!', file=file)
    print('Another line', file=file)
