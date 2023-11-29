# FIlename: test1.py

import decimal
import sys

var = '你是谁？？'
print(f'param = {var}')

# var = input('请输入：')
# print(f'var = {var}，type(var) = {type(var)}')
# var = int(var)
# print(f'var = {var}，type(var) = {type(var)}')


encodings = sys.getdefaultencoding()
print(f'encodings = {encodings}')

print('8*6=',8*9)

def abs(a):
    if a > 0:
        return a
    else:
        return -a

print(f'abs(a) = {abs(6)}')
print(f'abs(a) = {abs(-6)}')

print(123)
print(1.23)
print(0.123)
print(1.23e5)
print(123e3)
print(123e5)
print(123e-5)

money = decimal.Decimal('123.456')
print(f'money = {money}')


# 转义
print('''
line1 ,
line2...
''')

print(r'\\\t\\')


# 布尔值 True，False


'''
空值 Node，py中没有像java中的null，py中使用None来代替
None不能理解为0，因为0是有意义的，而None是一个特殊的空值。
'''
# print(null)
print(None)

'''
py中的变量，py中不像Java里面有强制类型，也没有类型声明，任何一个数据类型都可以存储在变量里面
'''
var = 123
print(var)
var = 'sac255'
print(var)
var = True
print(var)


# 想想下面的输出结果
a = 'aaa'
b = a
a = 'qqq'
print(f'a = {a}, id(a) = {id(a)}') # a = qqq, id(a) = 2630013064624
print(f'b = {b}, id(b) = {id(b)}') # b = aaa, id(b) = 2630013064752

# 可以得出结论，在py中是赋值，是深拷贝，而不是地址。

list1 = [1,2,3]
list2 = list1
print(f'list1 = {list1}, id(list1) = {id(list1)}') # list1 = [1, 2, 3], id(list1) = 2395989340480
print(f'list2 = {list2}, id(list2) = {id(list2)}') # list2 = [1, 2, 3], id(list2) = 2395989340480

list1.append(666)
print(f'list1 = {list1}, id(list1) = {id(list1)}') # list1 = [4, 5, 6, 666], id(list1) = 1879436242112
print(f'list2 = {list2}, id(list2) = {id(list2)}') # list2 = [1, 2, 3, 666], id(list2) = 2395989340480

list1 = [4,5,6]
print(f'list1 = {list1}, id(list1) = {id(list1)}') # list1 = [4, 5, 6], id(list1) = 1879436242112
print(f'list2 = {list2}, id(list2) = {id(list2)}') # list2 = [1, 2, 3], id(list2) = 2395989340480

# 根据以上可以得出，对于基本数据类型是值传递，而对象类型是地址传递


'''
常量
    所谓常量就是不能变的变量，比如常用的数学常数π就是一个常量。在Python中，通常用全部大写的变量名表示常量：
    注意，在py中没有关键字类限制常量不可被修改，因此，约定俗成对于全大写加下划线的变量尽量不要去修改他的值
'''
PI = 3.1415926
print(PI)


'''py中的除法
    浮点除法： 使用一个除号 /
    整数除法： 使用两个除号 //
'''
print(10 / 3)
print(10 // 3)


age = 55 #int(input('输入你的年龄：'))
if age > 18:
    print('大于18')
else:
    print('小于等于18')


# 字符串

str = 'hello world'
print(type(str))
print(id(str))

print(2**16)

print('=================')
A = 'A'
num65 = 65
print(ord(A))
# print(chr(A))
# print(ord(num65))
print(chr(num65))

str = '你' #只能单个字符
print(ord(str))
ordd = ord(str)
print(chr(ordd))

print(b'ABC'.decode('ascii'))


'''格式化输出 格式化
在Python中，采用的格式化方式和C语言是一致的，用%实现
%d	整数
%f	浮点数
%s	字符串
%x	十六进制整数
'''
print('亲爱的xxx你好！你xx月的话费是xx，余额是xx')

print('亲爱的%s你好！你%d月的话费是%f，余额是%f' % ('张三', 6, 63.3, 96.3))

print('%d转义百分号 %%' % (666))
# print('%d转义百分号 %' % (666)) #ValueError: incomplete format


'''format()格式化输出，使用{}来做占位符，使用数字从0开始来做下表，并且:后面的.2f指定了格式化参数（即保留两位小数）'''

print('亲爱的{0}你好！你{1}月的话费是{2}，余额是{3:.5f}'.format('张三', 6, 63.3, 96.3156156156))

''' r'指定字符串的输出' '''
r = 36
PI = 3.141592666666
print(fr'输出字符串 {r}， {PI:.3f}，{PI}')


aa = 'abc'
bb = aa.replace('b', 'B')
print(aa)
print(bb)

