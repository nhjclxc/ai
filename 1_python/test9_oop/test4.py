#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/1 20:40
# File      : test4.py
# Project   : 1_python
# explain   : 获取对象信息



class Student:
    # 下面的school是类属性
    school = '来自的学校'
    def __init__(self, name='None', age=18):
        # 下面的name，age是实列属性
        self.name = name
        self.age = age

    def to_string(self):
        return f'Student = {{name = {self.name}, age = {self.age}}}'


stu = Student('张三')
print(stu.to_string())

# 访问实列属性
print(stu.name)
# 访问类属性
print(Student.school)


print(type(None))
print(type(abs))
print(type(stu))
print(type(stu.name))
print(type(stu.to_string))
print(type(type(stu.to_string)))
print(type(type(type(stu.to_string))))

# 如果要获得一个对象的所有属性和方法，可以使用dir()函数，它返回一个包含字符串的list，
print(dir(stu))

len((1,2,3,'3'))

'''
仅仅把属性和方法列出来是不够的，配合 getattr()、setattr()以及hasattr()，我们可以直接操作一个对象的状态：
    getattr()：是获取对象属性对应的值
    setattr()：设置对象属性对应的值
    hasattr()：判断是否有某一个属性 
'''
print(getattr(stu, 'name'))
setattr(stu, 'name', '你是张三？')
print(getattr(stu, 'name'))
print(hasattr(stu, 'na22me'))
print(hasattr(stu, 'name'))



# 为了统计学生人数，可以给Student类增加一个类属性，每创建一个实例，该属性自动增加：
class Student2(object):
    count = 0

    def __init__(self, name):
        self.name = name
        Student2.count += 1


# 测试:
if Student2.count != 0:
    print('测试失败!')
else:
    bart = Student2('Bart')
    if Student2.count != 1:
        print('测试失败!')
    else:
        lisa = Student2('Bart')
        if Student2.count != 2:
            print('测试失败!')
        else:
            print('Student2:', Student2.count)
            print('测试通过!')
