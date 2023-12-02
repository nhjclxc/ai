#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author    : LuoXianchao
# Datetime  : 2023/12/1 20:14
# File      : test3_base_supper.py
# Project   : 1_python
# explain   : 继承和多态


class Animal():
    def __init__(self):
        pass

    def run(self):
        print(f'{self.__class__.__name__} is runing')

animal = Animal()

class Dog(Animal):
    def __init__(self):
        super().__init__()

dog = Dog()
dog.run()
print(isinstance(dog, Dog))
print(isinstance(dog, Animal))

class Cat(Animal):
    def __init__(self):
        super().__init__()

cat = Cat()
cat.run()
print(isinstance(cat, Cat))
print(isinstance(cat, Animal))


# 测试多态
def test_animal_run(animal):
    animal.run()

test_animal_run(animal)
test_animal_run(dog)
test_animal_run(cat)


