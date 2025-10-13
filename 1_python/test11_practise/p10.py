class Person:

    def __init__(self, name, addr=None, age=18):
        # 普通名称，没有下划线。
        # 可见性：公开（public）。
        # 任何地方都可以读写，外部可自由访问。
        self.name = name

        # 单下划线 _ 开头是 约定俗成的“受保护属性”。
        # 可见性：Python 并不会阻止访问，外部仍然可以访问：只是提醒使用者：这是内部实现细节，不建议外部直接访问。
        # 用途：用作类内部或子类访问，外部尽量不要直接使用。
        self._addr = addr

        # 双下划线 __ 开头是 名称改写（name mangling）机制。
        # 作用：Python 会把 __age 改名为 _ClassName__age，例如：_Person__age
        # 可见性：外部不能直接通过p.__age访问，会报错：AttributeError: 'Person' object has no attribute '__age'
        # 用途：防止子类或外部误用（伪私有）。
        # 注意：这不是完全私有，只是“伪私有”，防止名字冲突。
        self.__age = age

    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def get_age(self):
        return self.__age

    def __str__(self):
        # 子啊f""格式化字符串中 {{ 转义为 {，   }} 转义为 }
        return f"Person {{name = {self.name}, _addr = {self._addr}}}."


def test1():
    #   可见性和属性装饰器

    p1 = Person("zhangsan", "住北京", 18)

    print(p1.name)
    print(p1._addr)
    # __age 前面两个下划线__表示这个属性是外部不可见的
    # print(p1.__age)  # AttributeError: 'Person' object has no attribute '__age'
    print(p1._Person__age)  # 18 # 可以访问，但不建议
    print(p1.get_age())
    print(p1)

    pass


def test2():
    #   动态属性和动态方法
    p1 = Person("zhangsan", "住北京", 18)
    print(p1)
    p2 = Person("wangwu", "住上海", 28)
    print(p2)

    # 通过‘实例对象.属性名’的方式动态给实例对象添加属性，只有当前实例才有动态添加的属性，其他实例没有
    # print(p1.sex)  # AttributeError: 'Person' object has no attribute 'sex'
    p1.sex = 1
    print(p1.sex)

    # print(p2.sex)  # AttributeError: 'Person' object has no attribute 'sex'

    def say_hello():
        print("hello")

    # 动态方法
    p2.say = say_hello
    # p1.say()  # AttributeError: 'Person' object has no attribute 'say'
    p2.say()

    # 如果不希望在使用对象时动态的为对象添加属性，可以使用 Python 语言中的__slots__魔法。
    # 对于 Person 类来说，可以在类中指定__slots__ = ('name', 'age')，这样 Person 类的对象只能有name和age属性，

    pass


class Triangle:

    # 类属性 count，用于统计创建类多少个三角形实例了
    count = 0

    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
        Triangle.count += 1

    # 对象方法、类方法、静态方法都可以通过“类名.方法名”的方式来调用，区别在于方法的第一个参数到底是普通对象还是类对象，还是没有接受消息的对象。静态方法通常也可以直接写成一个独立的函数，因为它并没有跟特定的对象绑定。

    # 使用staticmethod装饰器声明了is_valid方法是Triangle类的静态方法
    @staticmethod
    def is_valid(a, b, c):
        """判断三条边长能否构成三角形(静态方法)"""
        return a + b > c and b + c > a and a + c > b

    @classmethod
    def get_count(cls):
        return cls.count

    def perimeter(self):
        """计算周长"""
        return self.a + self.b + self.c

    # 使用py内置的 property 装饰器将area方法变成一个属性，这样就可以点出属性了，不需要掉方法
    @property
    def area(self):
        """计算面积"""
        p = self.perimeter() / 2
        return (p * (p - self.a) * (p - self.b) * (p - self.c)) ** 0.5


def test3():
    #   静态方法和类方法
    print(Triangle.is_valid(3, 4, 5))
    print(Triangle.is_valid(3, 4, 3))

    t = Triangle(3, 4, 5)
    print(t.area)
    # print(t.area())  # TypeError: 'float' object is not callable
    print(t.perimeter())

    print(Triangle.get_count())
    Triangle(3, 4, 5)
    Triangle(3, 4, 5)
    print(Triangle.get_count())
    Triangle(3, 4, 5)
    print(Triangle.get_count())

    #  总结
    # 静态方法 = 类里的普通函数
    # 类方法 = 绑定类的函数，可访问类属性
    # 实例方法（普通方法） = 绑定实例，可访问实例属性和类属性


class Person2:
    """人"""

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def eat(self):
        print(f'{self.name}正在吃饭.')

    def sleep(self):
        print(f'{self.name}正在睡觉.')


class Student(Person2):
    """学生"""

    def __init__(self, name, age):
        super().__init__(name, age)

    def study(self, course_name):
        print(f'{self.name}正在学习{course_name}.')


class Teacher(Person2):
    """老师"""

    def __init__(self, name, age, title):
        super().__init__(name, age)
        self.title = title

    def teach(self, course_name):
        print(f'{self.name}{self.title}正在讲授{course_name}.')


def test5():
    #   继承和多态

    stu1 = Student('白元芳', 21)
    stu2 = Student('狄仁杰', 22)
    tea1 = Teacher('武则天', 35, '副教授')
    stu1.eat()
    stu2.sleep()
    tea1.eat()
    stu1.study('Python程序设计')
    tea1.teach('Python程序设计')
    stu2.study('数据科学导论')

    pass


if __name__ == "__main__":
    # 面向对象 进阶
    #   可见性和属性装饰器
    #   动态属性和动态方法
    #   静态方法和类方法
    #   继承和多态

    # test1()

    # test2()

    # test3()

    test5()
