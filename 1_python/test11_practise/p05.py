# /前面的参数是强制位置参数，只能通过位置传参，不能使用关键字传参
def make_judgement1(a, b, c, /):
    return a + b > c and b + c > a and a + c > b


# *后面的参数是命名关键字参数，只能通过关键字传参，不能使用位置传参
def make_judgement2(*, a, b, c):
    return a + b > c and b + c > a and a + c > b


def demo(a, b, /, c, d, *, e, f):
    """
    | 参数位置         | 调用方式       |
    | ------------ | ---------- |
    | `/` 之前       | 只能用位置参数    |
    | `/` 与 `*` 之间 | 既可位置，也可关键字 |
    | `*` 之后       | 只能用关键字参数   |
    """
    print(a, b, c, d, e, f)


def p03_01():
    print(make_judgement1(3, 4, 5))
    # TypeError: make_judgement1() got some positional-only arguments passed as keyword arguments: 'a, b, c'
    # print(make_judgement1(a=3 ,b=4 ,c=5))

    # TypeError: make_judgement2() takes 0 positional arguments but 3 were given
    # print(make_judgement2(3,4,5))

    print(make_judgement2(a=3, b=4, c=5))
    print(make_judgement2(b=3, c=4, a=5))

    # demo(11,22, 33,44,55,66)
    demo(11, 22, 33, 44, e=55, f=66)
    demo(11, 22, 33, 44, e=55, f=66)
    demo(11, 22, d=33, c=44, e=55, f=66)

    pass


def add(a=0, b=0):
    # 默认参数
    return a + b

def demo_args(*args):
    print(args)
    for arg in args:
        print(arg)

# 如果我们希望通过“参数名=参数值”的形式传入若干个参数，具体有多少个参数也是不确定的，我们还可以给函数添加可变关键字参数，把传入的关键字参数组装到一个字典中，代码如下所示。

# 参数列表中的**kwargs可以接收0个或任意多个关键字参数
# 调用函数时传入的关键字参数会组装成一个字典（参数名是字典中的键，参数值是字典中的值）
# 如果一个关键字参数都没有传入，那么kwargs会是一个空字典
def foo(*args, **kwargs):
    print(args)
    print(kwargs)


if __name__ == "__main__":
    # p03_01()

    print(add())
    print(add(1,2))
    demo_args()
    demo_args(1,2,3)
    demo_args(1,2,3,5,6,7,)

    foo(3, 2.1, True, name='骆昊', age=43, gpa=4.95)

    pass
