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

