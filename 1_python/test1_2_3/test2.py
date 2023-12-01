# FIlename: test2.py

'''
字典 dict = {'key': value}、列表 list = [v1, v2, v3...]、元组 tuple = (v1, v2, v3...)

`list`、`tuple`、`dict`和`set`是不同的数据类型，它们之间不能直接进行转换。如果需要将它们之间进行转换，需要使用相应的转换函数，
    如`list()`、`tuple()`、`dict()`和`set()`
'''

def dict_test():
    '''
        需要牢记的第一条就是dict的key必须是不可变对象
    '''
    dict = {'key1': 'value1'}
    print(dict)
    # 添加元素
    dict['key2'] = 666
    dict.update({'key3' : 555, 'key4' : "value4"})# 同时添加多个元素
    print(dict)
    # 移除指定元素
    print(dict.pop('key3'))
    print(dict)
    # 获取指定key对应的值
    print(dict.get('key3'))
    print(dict.get('key2'))
    # 获取字典的键值对列表
    print(dict.items())
    print('遍历')
    # 遍历 字典
    for k, v in dict.items():
        print(f'k = {k}，v = {v}')


    tt = (1,2,3)
    tt2 = (1,2,3,[1,2,3])
    ll = [1,2,3]
    dict = {}
    dict[tt] = 'tuple'
    # dict的key只能是不变对象，而list是可变对象所以西面保存，而上面不会保存
    # dict[ll] = 'list' # TypeError: unhashable type: 'list'
    # dict[tt2] = 'tuple-list' # TypeError: unhashable type: 'list'
    print(dict)

def set_test():
    '''
        set不能重复  set = {v1, v1, v1, v3 ...}
    '''
    set1 = {1,2,3,4,5,5,5,}
    print(set1)
    print(len(set1))

    # set2 = {1,2,3, [1,2,3]}
    # set3 = {1,2,3, [1,2,3]}
    # print(id(set3),id(set2))

    # 交集 并集 差集，对称差集，子集，超级
    s1 = {0,1,2,3,4,5,6}
    s2 = {4,5,6,7,8,9}
    # 交集
    print(s1.intersection(s2))
    print(s1 & s2)
    # 并集
    print(s1.union(s2))
    print(s1 | s2)
    # 差集 s1 - s2，去除s1中在s2里面出现过的元素
    print(s1.difference(s2))
    print(s1 - s2)
    # 对称差集
    print(s1.symmetric_difference(s2))
    print(s1 ^ s2)
    # {1,2,3}是否是s1的子集，也就是说{1,2,3}里面的所有元素是不是都被s1包含了
    print({1,2,3}.issubset(s1))
    # s1是否是{1,2,3}的超级
    print(s1.issuperset({1,2,3}))

    pass

def list_test():
    ''' 列表 list = [v1, v2, v3...]
        list是一种有序的集合，可以随时添加和删除其中的元素。
    '''
    list = ['v1', 'v2', 'v3', 25]
    print(list)
    # 追加
    list.append(1856)
    print(list)
    # 指定位置插入元素
    list.insert(2, 88888)
    print(list)
    # 将指定位置的元素替换
    list[1] = 666
    print(list)
    # 获取指定元素的索引
    print(list.index(88888))
    print(list.index(88888, 0, 5))
    # print(list.index(99999)) #要查找的元素不在列表时，抛出以下异常，如果是在列表里面查找指定元素并且返回对应索引的话就要加异常捕获了 ValueError: 99999 is not in list
    print(find_index(list, 88888))
    print(find_index(list, 99999))
    # 移除最后一个，返回出栈的元素
    print(list.pop())
    # 移除指定位置的元素
    print(list.pop(2))
    print(list)
    # 在列表中移除指定的值
    list.remove(25)
    print(list)
    # 移除一个已经不在列表中的元素
    # list.remove(25) #ValueError: list.remove(x): x not in list
    # print(list)
    # 列表获取长度
    l = len(list)
    print(l)
    # 按照索引访问
    print(list[0])
    print(list[1])
    print(list[2])
    # print(list[3]) # IndexError: list index out of range
    # 从后往前访问，最后一个是-1，不断减一
    print(list[-1])
    print(list[-2])
    print(list[-3])
    '''
        v1  v2  v3  v4  v5
    正   0   1   2   3   4
    反   -5  -4  -3  -2  -1    
        由此可得出list[]列表的底层实现是：列表（List）的底层实现是一个动态数组（Dynamic Array）。Python 的列表是一个可变长度的数据结构，可以存储任意类型的元素，并且支持动态调整大小
    '''

    '''
        类似于二维数组，一个列表里面还有一个列表，想要操作内部列表的时候获取到，之后就是和一维列表一样操作了
    '''
    p0 = ['福州', '泉州', '厦门', '莆田']
    p1 = ['杭州', '宁波', '诸暨', '绍兴']
    p2 = ['合肥', '安庆', '芜湖', '亳州']
    p = [p0, p1, p2]
    print(p0)
    print(p1)
    print(p)
    # '莆田' -> '沙县'
    pp = p[0]
    print(pp)
    pt_ind = find_index(pp, '莆田')
    if pt_ind != None:
        pp[pt_ind] = '沙县'
    print(pp)

    # list排序
    print(pp)
    pp.sort()
    print(pp)

    print()

def find_index(lst, element):
    try:
        index = lst.index(element)
        return index
    except ValueError:
        return None  # 如果元素不在列表中，返回 None 或者其他你认为合适的值

def tuple_test():
    '''
        元组 tuple = (v1, v2, v3...)
        tuple和list非常类似，但是tuple一旦初始化就 不 能 修 改，相当于在Java里面的变量加了finnal关键字
    '''
    tuple1 = ('张三', '里斯', '撒擦啊是')
    print(tuple1.index('张三'))
    # print(tuple1.index('张s三'))

    t2 = (666) # 这个是什么？ 数字666还是元组(666)？
    print(f't2 = {t2}, type(t2) = {type(t2)}') #可以看出是一个数字<class 'int'>
    # 一下是定义一个元组变量的方法，就是在后面加一个逗号,
    t3 = (666, )
    print(f't3 = {t3}, type(t3) = {type(t3)}') # <class 'tuple'>

    # 二维tuple和二维列表一样


    tt = ([123, 456, 789], 111)
    print(tt)
    ll = tt[0]
    print(ll)
    ll.append(666)
    print(ll)

    pass

'''
列表list和元组tuple有什么区别
'''
def list_tuple_test():
    tl = ('v0',('v10','v11','v12'), ['v20','v21','v22'])
    print(tl)
    # tl[0] = 'sss' # TypeError: 'tuple' object does not support item assignment
    tl1 = tl[1]
    # tl1[1] = 'qqq' # TypeError: 'tuple' object does not support item assignment
    tl2 = tl[2]
    print(tl2)
    tl2.append('666')
    print(tl2)

    ss = ('莉莉', '晴晴', '圆圆', ['小明', '小红', ['小粉', '小鬼'], '壮壮'])
    print(ss)
    ss[3][2].pop()
    print(ss)
    ss[3].pop(2)
    print(ss)

    pass

def slice_test():
    '''切片操作，是一个前开后闭的切片操作 ( ]
    [start = 0 : end = len-1 : step = 1]
        start默认为第一个开始，不包含
        end默认到最后一个，包含
        step切片步长默认为1
    '''
    my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    print(my_list[1 : 6 : 1])
    print(my_list[1 : 6 : 2])
    print(my_list[3 : ])
    print(my_list[ : 3])
    print(my_list[ : ])  # 原始列表，表示了
    # 利用步长-1，将list进行逆置
    print(my_list[ : : - 1])  # [9, 8, 7, 6, 5, 4, 3, 2, 1]


    my_tuple = (1, 2, 3, 4, 5, 6, 7, 8, 9)
    print(my_tuple[1 : 6 : 1])
    print(my_tuple[1 : 6 : 2])
    print(my_tuple[3 : ])
    print(my_tuple[ : 3])
    print(my_tuple[ : ])  # 原始列表，表示了
    # 利用步长-1，将list进行逆置
    print(my_tuple[ : : - 1])  # [9, 8, 7, 6, 5, 4, 3, 2, 1]


    pass

if __name__ == '__main__':
    # dict_test()
    set_test()
    # list_test()
    # tuple_test()
    # list_tuple_test()

    # slice_test()


    # pp = ['福州', '泉州', '厦门', '莆田']
    # print(pp)
    # pp.sort()
    # print(pp)


    pass

