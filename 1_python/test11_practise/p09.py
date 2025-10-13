class Student:
    """学生类"""

    # 在我们调用Student类的构造器创建对象时，首先会在内存中获得保存学生对象所需的内存空间，
    # 然后通过自动执行__init__方法，完成对内存的初始化操作，也就是把数据放到内存空间中。

    # 初始化方法
    def __init__(self):
        # 对象的属性声明必须在__init__方法里面执行
        self.name = '不愿透露姓名的一个学生'

    def study(self, course_name):
        self.course_name = course_name
        print(f'学生正在学习{course_name}.')

    def play(self):
        print(f'学生正在玩游戏.')


def test1():
    global s3
    s1 = Student()
    s1.study('课程1')
    s1.play()
    s2 = Student()
    s3 = s2
    print(s1)
    print(s2)
    print(s3)
    print(id(s1), id(s2), id(s3))
    print(hex(id(s1)), hex(id(s2)), hex(id(s3)))
    # 根据以上可以知道，s3=s2这个赋值语句是一个地址的赋值
    print(s2.name, s3.name)  # 不愿透露姓名的一个学生 不愿透露姓名的一个学生
    s3.name = "一个转校生"
    print(s2.name, s3.name)  # 一个转校生 一个转校生
    # 根据以上也可以知道，s3=s2这个赋值语句是一个地址的赋值
    # 对象实例.实例方法(参数列表)这是一种普通通用的对象方法调用方式，如s1.study('课程1')
    # 此外py还有一种方法，即：类名.方法名(实例对象, 参数列表)
    Student.study(s2, "～～课程课程！！")

class Point:
    def __init__(self, x, y):
        self.__x = x
        self.__y = y

    # 运算符重载 “+”，使用p1+p2时默认调用这个方法
    def __add__(self, other):
        self.__x += other.__x
        self.__y += other.__y

    # 返回新对象，不修改原数据，建议使用不修改原对象的方法，如果定义这个方法又想修改原对象的花，可以使用 p1 += p2 的方式达到目的
    def __add2__(self, other):
        return Point(self.__x + other.__x, self.__y + other.__y)

    # 使用 += 运算进行原地修改，__iadd__可以修改原对象
    def __iadd__(self, other):
        self.__x += other.__x
        self.__y += other.__y
        return self

    def distance(self, other):
        return pow((self.__x - other.__x) ** 2 + (self.__y - other.__y) ** 2, 0.5)

    def __str__(self):
        return f'({self.__x}, {self.__y})'


"""
1️⃣ Python 的运算符重载
Python 中的运算符本质上都是**魔法方法（special methods）**的语法糖。

| 运算符  | 对应魔法方法                |
| ---- | --------------------------- |
| `+`  | `__add__(self, other)`      |
| `-`  | `__sub__(self, other)`      |
| `*`  | `__mul__(self, other)`      |
| `/`  | `__truediv__(self, other)`  |
| `//` | `__floordiv__(self, other)` |
| `%`  | `__mod__(self, other)`      |
| `**` | `__pow__(self, other)`      |
| `+=` | `__iadd__(self, other)`     |
| `-=` | `__isub__(self, other)`     |
| `==` | `__eq__(self, other)`       |
| `<`  | `__lt__(self, other)`       |
| `>`  | `__gt__(self, other)`       |

"""

def test2():
    # 定义一个Point类，计算点与点之间的距离
    p1 = Point(3, 5)
    p2 = Point(6, 9)
    print(p1)  # 调用对象的__str__魔法方法
    print(p2)
    print(p1.distance(p2))

    print(p1)
    p1.__add__(p2)
    print(p1)
    p1 + p2
    print(p1)

    print(p1)
    p1 += p2
    print(p1)


    # p1 - p2 # TypeError: unsupported operand type(s) for -: 'Point' and 'Point'


    pass


if __name__ == "__main__":
    # 面向对象 入门

    # test1()
    test2()

