def fun1():
    L = []
    n = 1
    while n <= 20:
        L.append(n)
        n = n + 2
    print(L)

    # 切片，取L的前一半
    i = 0
    slice_L = []
    while i < len(L) // 2:
        slice_L.append(L[i])
        i += 1
    return slice_L


def slice_test():
    '''切片操作，是一个前开后闭的切片操作 ( ]
    [start = 0 : end = len-1 : step = 1]
        start默认为第一个开始，不包含
        end默认到最后一个，包含
        step切片步长默认为1
    '''
    my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    print(my_list[1: 6: 1])
    print(my_list[1: 6: 2])
    print(my_list[3:])
    print(my_list[: 3])
    print(my_list[:])  # 原始列表，表示了
    # 利用步长-1，将list进行逆置
    print(my_list[:: - 1])  # [9, 8, 7, 6, 5, 4, 3, 2, 1]

    my_tuple = (1, 2, 3, 4, 5, 6, 7, 8, 9)
    print(my_tuple[1: 6: 1])
    print(my_tuple[1: 6: 2])
    print(my_tuple[3:])
    print(my_tuple[: 3])
    print(my_tuple[:])  # 原始列表，表示了
    # 利用步长-1，将list进行逆置
    print(my_tuple[:: - 1])  # [9, 8, 7, 6, 5, 4, 3, 2, 1]

    pass


def trim1(s):
    '''利用切片操作，实现一个trim()函数，去除字符串首尾的空格'''
    print(f' len = {len(s)}')
    while s.startswith(' '):
        s = s[1:]
    while s.endswith(' '):
        s = s[: -1] #倒数第二个往前面截取
    print(f' len = {len(s)}')

    return s


def trim(s):
    '''利用切片操作，实现一个trim()函数，去除字符串首尾的空格'''
    return trim0(s, ' ')

def trim0(s, flag = ' '):
    '''利用切片操作，实现一个trim()函数，去除字符串首尾的空格'''
    if s.startswith(flag):
        return trim0(s[1:], flag)
    elif s.endswith(flag):
        return trim0(s[: -1], flag) #倒数第二个往前面截取
    return s


def strip(s, flag):
    return trim0(s, flag)



if __name__ == '__main__':
    # print(fun1())
    print(trim('  asd '))
    if trim('hello  ') != 'hello':
        print('测试失败!')
    elif trim('  hello') != 'hello':
        print('测试失败!')
    elif trim('  hello  ') != 'hello':
        print('测试失败!')
    elif trim('  hello  world  ') != 'hello  world':
        print('测试失败!')
    elif trim('') != '':
        print('测试失败!')
    elif trim('    ') != '':
        print('测试失败!')
    else:
        print('测试成功!')
