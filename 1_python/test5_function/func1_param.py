# Filename: func1_param.py

from arrow import now


# 详解Python函数参数定义及传参（必备参数、关键字参数、默认可省略参数、可变不定长参数、*args、**kwargs）
# https://blog.csdn.net/yyykj/article/details/103122665

# Python中函数参数定义及调用函数时传参大体可分必备参数、关键字参数、默认可省略参数、不定长元组参数、不定长关键字参数等


# 1.必备参数__仅赋值传参
def fun1(name, age, addr, hobby):
    print(f'name = {name}，age = {age}，addr = {addr}，hobby = {hobby}')

fun1('zhangsan', 23, '你的地址','吃喝玩乐')

fun1('你的地址', 'zhangsan','吃喝玩乐','23')  # ？？？？？


# 2.必备参数__键值对传参（关键字参数）
fun1(addr='你的地址', name='zhangsan', hobby= '看斗鱼', age=23)


# 3.默认可省略参数
def fun3(name, age, addr, hobby='看抖音'):
    print(f'name = {name}，age = {age}，addr = {addr}，hobby = {hobby}')

fun3(addr='你的地址', name='zhangsan', hobby= '看斗鱼', age=23)
fun3(addr='你的地址', name='zhangsan', age=23)


# def fun4(name, age, addr='北京', hobby):
#     """ 提示非默认参数不能在默认参数之后，代码不能正常执行。  """
#     print(f'name = {name}，age = {age}，addr = {addr}，hobby = {hobby}')


# 字典 dict = {'key' : value}、列表 list = [v1,v2,v3...]、元组tuple = (v1,v2,v3...)
# 5.字典参数 dict = {'key' : value}
def fun5(dict):
    print(f"dict['key1'] = {dict['key1']}")
    print(f"dict['key2'] = {dict['key2']}")
    print(f"dict['key3'] = {dict['key3']}")
    print('\n')

fun5({'key1' : 'asd','key2' :  2345,'key3' :  now()})
# fun5({'key1' : 'asd','key2' :  2345,'qqqqq' :  now()}) # KeyError: 'key3'


# 6.列表参数 list = [v1,v2,v3...]
def fun6(list):
    print(f'list[0] = {list[0]}')
    print(f'list[1] = {list[1]}')
    print(f'list[2] = {list[2]}')
    print('\n')

fun6(['asd', 2345, now()])


# 7.元组参数 tuple = (v1,v2,v3...)
def fun7(tuple):
    print(f'tuple[0] = {tuple[0]}')
    print(f'tuple[1] = {tuple[1]}')
    print(f'tuple[2] = {tuple[2]}')
    print('\n')

fun7(('asd', 2345, now()))


# 8.不定长元组参数（*args） ，类似于Java里面的可变参数
def fun8(i, *params):
    print(f'i = {i}')
    for param in params:
        print(param)
    print('\n')

fun8(666,'zhangsan', 23, '你的地址','吃喝玩乐')
fun8(666, 888, 'zhangsan', 23, '你的地址','吃喝玩乐')
fun8(666, [888, 'zhangsan', 23, '你的地址','吃喝玩乐']) # []被当成整体
fun8(666, (888, 'zhangsan', 23, '你的地址','吃喝玩乐')) # ()当成整体


# 9.不定长字典参数（**kwargs）  keyword arguments
def fun9(**kwargs):
    """
    在函数定义中，**kwargs 会将传递给函数的关键字参数打包成一个字典（dictionary），其中关键字作为字典的键，对应的值作为字典的值。因此，kwargs 是一个代表关键字参数的字典变量。
    :param kwargs:
    :return:
    """
    for k, v in kwargs:
        print(f'k = {k}, v = {v}')
    for k, v in kwargs.items(): # 使用items方法来获取字典的键值对
        print(f'k = {k}, v = {v}')
    '''
    kwargs 是包含关键字参数的字典本身。(返回字典对象)
    kwargs.items() 是一个可迭代对象，它提供字典中键值对的视图，使你能够访问每个键值对的键和值。（返回字典对象里面的键值对）
    '''
    print(kwargs['k1'])
    print(kwargs['k2'])
    print(kwargs['k3'])

fun9(k1 = 'v11', k2 = 'v22', k3 = 'v33')


# python的参数传递是值传递
def fun10(**kw):
    if 'name' in kw:
        print(f"来源可变参数：name = {kw['name']}")
    if 'age' in kw:
        print(f"来源可变参数：age = {kw['age']}")

kw = {'name' : 'zhangsan', 'age' : 18, 'addr' : 'hk'}
# fun10(kw) # TypeError: fun10() takes 0 positional arguments but 1 was given
fun10(**kw)

def f1(a, b, c=0, *args, **kw):
    print('a =', a, 'b =', b, 'c =', c, 'args =', args, 'kw =', kw)

f1(1,2,3,kw='kw')
f1(a = 1, b = 2,kw='kw')


def mul(*args):
    '''可接收一个或多个数并计算乘积'''

    if len(args) == 0:
        raise TypeError('缺少参数')
    mul_ret = 1
    for a in args:
        mul_ret *= a
    return mul_ret

print('mul(5) =', mul(5))
print('mul(5, 6) =', mul(5, 6))
print('mul(5, 6, 7) =', mul(5, 6, 7))
print('mul(5, 6, 7, 9) =', mul(5, 6, 7, 9))
if mul(5) != 5:
    print('测试失败!')
elif mul(5, 6) != 30:
    print('测试失败!')
elif mul(5, 6, 7) != 210:
    print('测试失败!')
elif mul(5, 6, 7, 9) != 1890:
    print('测试失败!')
else:
    try:
        mul()
        print('测试失败!')
    except TypeError:
        print('测试成功!')

