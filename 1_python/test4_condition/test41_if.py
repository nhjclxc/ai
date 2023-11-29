# Filename: test41_if.py

score = 66
if score == 100:
    print('满分')
elif 80 <= score <= 99:
    print('优秀')
elif 60 <= score <= 79:
    print('及格')
else:
    print('不及格')

def print_level(score):
    if score == 100:
        print('满分')
    elif 80 <= score:
        print('优秀')
    elif 60 <= score:
        print('及格')
    else:
        print('不及格')


print_level(66)


# 类switch
def switcher_(arg):
    switcher = {
        1: '返回1',
        2: '返回2',
        3: '返回3',
    }
    return switcher.get(arg, '没见过的东西')


print(switcher_(1))
print(switcher_(66))


def match_(arg):
    # ret = None
    match arg:
        case 1:
            ret = '返回1'
        case 2 | 3 | 4 | 5 | 80:
            ret = '返回2'
        case arg if 50 < arg < 200:
            ret = '返回3'
        case _:  # 使用下划线_来表示未匹配上的结果
            # ret = '其他值 ' + str(arg)
            # ret = f'其他值 {arg}'
            ret = f'其他值 '.join(str(arg))
    return ret


print(match_(1))
print(match_(5))
print(match_(0))
print(match_(80))
print(match_(99))

def match_list(list):
    # 列表匹配
    match list:
        case ['c++']:
            print('有c++22')
        case ['c++', 'java']:
            print('有c++ java22')
        case _:
            print('没见过22')

match_list(['c++'])
match_list(['c++', 'java'])


# 使用一个分隔符来连接一个列表的所有元素成字符串
l3 = ['c++', 'java', 'py']
print('编程语言 '.join(l3))


'''三元表达式
    var = value_if_true if condition else value_if_false
'''
n = 5
var = n if n < 2 else n * 2
print(var)
