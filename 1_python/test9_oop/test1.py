#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/1 16:52
# File      : test1.py
# Project   : 1_python
# explain   : 测试py对象属性随时可以添加


class MyClass:
    def __init__(self):
        self.attribute1 = 'value1'
        self.attribute2 = 'value2'
        self.attribute3 = 'value3'

    def get_all_attributes(self):
        '''
            vars() 不传入参数时，返回当前作用域的属性字典。
            vars(object) 传入对象时，返回该对象的属性字典。
        '''
        return vars(self)  # 或者使用 return self.__dict__

    def to_string(self):
        dict = vars(self)
        str = self.__class__.__name__ + ' = {'
        dict_len = len(dict)
        i = 0
        for k, v in dict.items():
            str += k + ' = ' + v
            str += '' if dict_len-1 == i else ', '
            i += 1
        str += '}'
        return str

    def to_string2(self):
        attributes = ', '.join(f"{key} = {value}" for key, value in vars(self).items())
        return f"{self.__class__.__name__} = {{{attributes}}}"

if __name__ == '__main__':
    # py中类的属性可以随时的增加，
    # 因此，可以在创建实例的时候，把一些我们认为必须绑定的属性强制填写进去。
    # 而对于可能用到的属性，则可以动态添加

    # 创建对象实例
    obj = MyClass()
    # 调用方法获取所有属性并输出
    attributes = obj.get_all_attributes()
    for attribute, value in attributes.items():
        print(f"{attribute}: {value}")

    print()
    obj.name = '张三' #添加一个属性上去
    # 调用方法获取所有属性并输出
    attributes = obj.get_all_attributes()
    for attribute, value in attributes.items():
        print(f"{attribute}: {value}")

    print(obj.to_string())
    print(obj.to_string2())


    lst = [1, 2, 3, 4, 5, 6, 7]
    result = ',,'.join(str(item) for item in lst)
    print(result)

    pass



