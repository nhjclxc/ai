import copy
import random


def p03_01(n: int):
    # fib
    a, b = 0, 1
    if n == 1:
        return a
    if n == 2:
        return b
    for i in range(2, n + 1):
        a, b = b, a + b
    return b


def p03_02():
    # 将一个列表的数据复制到另一个列表中。
    lst1 = [1, 2, 3, 5, 6, 8, 9]
    print("lst1", lst1)

    lst2 = []
    for i in lst1:
        lst2.append(i)
    print("lst2", lst2)

    lst3 = lst1[:]
    print("lst3", lst3)

    lst5 = list.copy(lst1)
    print("lst5", lst5)

    lst6 = lst1.copy()
    print("lst6", lst6)

    lst8 = copy.copy(lst1)
    print("lst8", lst8)

    lst9 = [i for i in lst1]
    print("lst9", lst9)

    lst10 = lst1
    print("lst10", lst10)

    lst11 = lst1 * 1
    lst12 = lst1 * 2
    lst13 = lst1 * 3
    print("lst11", lst11)
    print("lst12", lst12)
    print("lst13", lst13)

    lst1[2] = 222

    print("lst1", lst1)
    print("lst2", lst2)
    print("lst3", lst3)
    print("lst5", lst5)
    print("lst6", lst6)
    print("lst10", lst10)
    pass


def p03_03():
    # 将一颗色子掷6000次，统计每种点数出现的次数

    countList = [0 for _ in range(6)]
    for i in range(0, 100):
        num = random.randrange(1, 7)
        countList[num - 1] = countList[num - 1] + 1

    for i in range(6):
        print(f"{(i + 1)} 出现的次数 {countList[i]}")

    print()

    # 使用enumerate来将列表组装成元组(index, value)返回
    for index, value in enumerate(countList):
        print(f"{(index + 1)} 出现的次数 {value}")


def p03_05():
    # 列表的运算
    lst1 = [1, 2, 3]
    lst2 = [6, 7, 8]
    # “+” 表示列表拼接
    # “*” num：表示将列表重复num次

    print("lst1", lst1)
    print("lst2", lst2)

    print(lst1 + lst2)
    print(lst1 * 2)  # 就是将list重复多少次

    print(6 in lst1)
    print(6 in lst2)
    print(6 not in lst1)
    print(6 not in lst2)

    print(lst1[0])
    print(lst1[1])
    print(lst1[2])
    print(lst1[-1])
    print(lst1[-2])
    print(lst1[-3])

    lst3 = [1, 2, 3]

    print(lst1 == lst3)
    print(lst1 > lst3)
    print(lst1 < lst3)
    print(lst1 != lst3)

    print(type(lst1))
    print(id(lst1))  # 输出变量地址
    print(id(lst3))

    pass


def p03_06():
    # match的使用，类似于Java的switch

    grade = 'A'

    match grade:
        case 'A':
            print("A等级")
        case 'B':
            print("B等级")
        case _:
            print("未知等级")

    score = 68

    if 100 > score >= 90:
        print("优")
    elif score >= 70:
        print("良")
    elif score >= 60:
        print("及格")
    else:
        print("不及格")

    # 用 match...case 的守卫 (guard)
    match score:
        case _ if 90 < score < 100:
            print("优")
        case _ if 70 < score:
            print("良")
        case _ if 60 <= score:
            print("及格")
        case _:
            print("不及格")

    pass


def p03_08():
    # 元组tuple的使用
    # 元组使用小括号()，元组不能改变，不支持修改
    # 列表使用中括号[]，列表可以改变，支持修改

    t1 = (1, 2, 3)
    t2 = (1)  # 注意：如果是一个元素的元组，元素后面必须接一个逗号,才会被编译器识别为元组，否则就是一个普通变量了
    t3 = (1,)
    tt = ()  # 表示空元组

    print(type(tt), tt)
    print(type(t1), t1)
    print(type(t2), t2)  # <class 'int'> 1
    print(type(t3), t3)  # <class 'tuple'> (1,)

    # 元组的打包和接包
    ta = 11, 22, 33  # 打包：将多个数据赋值给一个变量
    print(type(ta), ta)
    a1, a2, a3 = ta  # 解包：将一个元组赋值给多个变量
    print(type(a1), a1)
    # a1, a2, a3, aaa = ta  # 需要更多值才能解包 ValueError: not enough values to unpack (expected 4, got 3)

    t2 = (1, 2, 3, 4, 5, 6, 7, 8)
    # 如果要解包t2，那么就要8个变量，此时如果有100个元素的元组来了，就要定义100个变量了，显然不合理
    # 因此，py提供了“星号表达式”来接收多个变量
    t21, *t22, t23 = t2
    print(type(t21), t21)
    print(type(t22), t22)  # *来接收多个变量，使用变量的时候不要带*
    print(type(t23), t23)


def p03_09():
    # 交换变量的值
    # 在其他编程语言（除py）中，要想交换两个变量的值，那么必须引入第三个中间变量tmp才能实现两个变量的数据交换
    # 而在py中不需要，可以直接使用逗号表达式来交换
    a = 666
    b = 999
    print(f"a = {a}, b = {b}")
    a, b = b, a
    print(f"a = {a}, b = {b}")
    a, b = b - 111, a + 1  # 会先计算右边，计算玩结果之后再去赋值
    print(f"a = {a}, b = {b}")

    c = 123
    print(f"a = {a}, b = {b}, c = {c}")
    a, b, c = c, a, b
    print(f"a = {a}, b = {b}, c = {c}")

    # 元组与列表的互转
    tpl = (1, 2, 3)
    lst = [6, 7, 8]
    print(type(tpl), tpl)
    print(type(lst), lst)
    tpl, lst = tuple(lst), list(tpl)
    print(type(tpl), tpl)
    print(type(lst), lst)

    pass


def p03_10():
    # 字符串操作

    s1 = 'hello, world!'
    print(type(s1), s1)

    s2 = "hello"
    s3 = " "
    s5 = "world"
    ss = s2 + s3 + s5
    print(ss)

    print(s2 * 2)
    s22 = "hello"
    print(s2 == s22)
    print(s2 is s22)
    print(s2 is not s22)
    print(s2 in s22)
    print("hel" in s22)
    print("a", ord("a"))
    print("A", ord("A"))
    print("A" > "a")
    print("A" == "a")
    print("A" < "a")  # 比较ascii
    print(len(s22))
    print(s22[0])
    print(s22[1])

    # 字符串的遍历
    for i in range(len(s22)):
        print(i, s22[i])
    for item in s22:
        print(item)

    # 字符串操作常用的方法
    s111 = "  zzzxx  "
    print(s111)
    print(s111.strip())

    s = 'I love you'
    words = s.split()
    print(words)  # ['I', 'love', 'you']
    print('~'.join(words))  # I~love~you

    """
🧩 一、大小写相关
| 方法                 | 作用        | 示例                                      |
| ------------------ | --------- | --------------------------------------- |
| `str.upper()`      | 转为大写      | `'hello'.upper() → 'HELLO'`             |
| `str.lower()`      | 转为小写      | `'HELLO'.lower() → 'hello'`             |
| `str.title()`      | 每个单词首字母大写 | `'hello world'.title() → 'Hello World'` |
| `str.capitalize()` | 仅首字母大写    | `'python'.capitalize() → 'Python'`      |
| `str.swapcase()`   | 大小写互换     | `'PyThOn'.swapcase() → 'pYtHoN'`        |

🔍 二、查找与判断
| 方法                       | 作用                 | 示例                                 |
| ------------------------ | ------------------ | ---------------------------------- |
| `str.find(sub)`          | 返回子串索引（找不到返回 -1）   | `'apple'.find('p') → 1`            |
| `str.rfind(sub)`         | 从右边开始找             | `'apple'.rfind('p') → 2`           |
| `str.index(sub)`         | 类似 `find()`，但找不到报错 | `'apple'.index('p') → 1`           |
| `str.count(sub)`         | 统计子串出现次数           | `'banana'.count('a') → 3`          |
| `str.startswith(prefix)` | 判断是否以某前缀开头         | `'hello'.startswith('he') → True`  |
| `str.endswith(suffix)`   | 判断是否以某后缀结尾         | `'test.py'.endswith('.py') → True` |
| `in` 运算符                 | 判断子串是否存在           | `'a' in 'cat' → True`              |

🧮 三、判断字符类型（返回 True/False）
| 方法              | 作用     | 示例                               |
| --------------- | ------ | -------------------------------- |
| `str.isalpha()` | 全字母    | `'abc'.isalpha() → True`         |
| `str.isdigit()` | 全数字    | `'123'.isdigit() → True`         |
| `str.isalnum()` | 字母或数字  | `'abc123'.isalnum() → True`      |
| `str.isspace()` | 全空白字符  | `'   '.isspace() → True`         |
| `str.islower()` | 全小写    | `'abc'.islower() → True`         |
| `str.isupper()` | 全大写    | `'ABC'.isupper() → True`         |
| `str.istitle()` | 是否标题格式 | `'Hello World'.istitle() → True` |

✂️ 四、删除与替换
| 方法                               | 作用       | 示例                                 |
| -------------------------------- | -------- | ---------------------------------- |
| `str.strip()`                    | 去掉首尾空白字符 | `'  hi  '.strip() → 'hi'`          |
| `str.lstrip()`                   | 去掉左边空白   | `'  hi'.lstrip() → 'hi'`           |
| `str.rstrip()`                   | 去掉右边空白   | `'hi  '.rstrip() → 'hi'`           |
| `str.replace(old, new[, count])` | 替换子串     | `'aabb'.replace('a','x') → 'xxbb'` |

🔗 五、分割与拼接
| 方法                 | 作用        | 示例                                       |
| ------------------ | --------- | ---------------------------------------- |
| `str.split(sep)`   | 按分隔符切割成列表 | `'a,b,c'.split(',') → ['a','b','c']`     |
| `str.rsplit(sep)`  | 从右侧开始切割   | `'a,b,c'.rsplit(',',1) → ['a,b','c']`    |
| `str.splitlines()` | 按行切割      | `'a\nb\nc'.splitlines() → ['a','b','c']` |
| `sep.join(list)`   | 用分隔符连接列表  | `'-'.join(['a','b','c']) → 'a-b-c'`      |

🧱 六、对齐与填充
| 方法                              | 作用    | 示例                               |
| ------------------------------- | ----- | -------------------------------- |
| `str.center(width[, fillchar])` | 居中对齐  | `'hi'.center(6, '*') → '**hi**'` |
| `str.ljust(width[, fillchar])`  | 左对齐   | `'hi'.ljust(6, '-') → 'hi----'`  |
| `str.rjust(width[, fillchar])`  | 右对齐   | `'hi'.rjust(6, '-') → '----hi'`  |
| `str.zfill(width)`              | 左侧补 0 | `'42'.zfill(5) → '00042'`        |

🔡 九、翻转与查找技巧
| 方法 / 技巧                 | 作用        | 示例                    |
| ----------------------- | --------- | --------------------- |
| `str[::-1]`             | 翻转字符串     | `'abc'[::-1] → 'cba'` |
| `max(str)` / `min(str)` | 返回最大/最小字符 | `max('abcd') → 'd'`   |




    """

    pass


def p03_11():
    # 集合set，无序性、互异性、确定性
    # set里面保存不重复元素

    s = {1, 2, 3, 3, 2, 1}
    print(s)
    s.add(666)
    print(s)
    s.add(666)
    s.add(999)
    print(s)

    print(type(s), s)
    s = {}  # <class 'dict'> {}
    print(type(s), s)
    s = {1}  # <class 'set'> {1}
    print(type(s), s)
    s = {1, }  # <class 'set'> {1}
    print(type(s), s)

    print(set("hello"))

    s2 = {num for num in range(10)}
    print(s2)

    for num in s2:
        print(num)
    print()
    for index, val in enumerate(s2):
        print(index, val)

    # 集合的运算
    # Python 为集合类型提供了非常丰富的运算，主要包括：成员运算、交集运算、并集运算、差集运算、比较运算（相等性、子集、超集）等。

    a = s2.pop()
    a = s2.pop()
    print(a)
    a = s2.remove(2)
    print(a)
    a = s2.discard(2)
    print(a)

    pass


def p03_12():
    # 字典
    # key-value结构，key唯一
    data = {
        "name": "zhangsan",
        "age": 18,
        "addr": "北京"
    }
    print(data)
    print(data.get("name"))
    print(data["age"])

    # dict函数(构造器)中的每一组参数就是字典中的一组键值对
    d2 = dict(name="zhangsan", age=18)
    print(d2)

    # # 可以通过Python内置函数zip压缩两个序列并创建字典
    m1 = dict(zip("Abcd", "1235"))
    print(m1)

    m2 = {index: index * 2 for index in range(10)}
    print(m2)
    print(len(m2))  # 输出有多少个key

    # 字典套字典
    school = {
        "name": "学校1",
        "addr": "学校地址",
        "students": [
            {"name": "张三", "age": 18, "address": "aa"},
            {"name": "里斯", "age": 28, "address": "bb"},
        ]
    }

    print(school)

    print("tel" in school)
    print("name" in school)
    print("name" in school.keys())

    # 字典遍历
    for key, value in school.items():
        # if type(data) == dict or type(data) == list:
        if isinstance(value, dict) or isinstance(value, list):
            for value2 in value:
                for key3, value3 in value2.items():
                    print(f"\t key3: {key3}, value3: {value3}")
        else:
            print(f"key: {key}, value: {value}")

    print(school.get("name"))
    print(school.get("name", 11))
    print(school.get("name1"))
    print(school.get("name1", 11))

    name = school.pop("name")
    print(name)
    x = school.popitem()
    print(x)

    del school["name"]



    pass


def p03_13():

    # 使用dict统计一段字符串每一个字符出现的次数
    str1 = "Man is distinguished, not only by his reason, but by this singular passion from other animals, which is a lust of the mind, that by a perseverance of delight in the continued and indefatigable generation of knowledge, exceeds the short vehemence of any carnal pleasure."

    countMap = {}
    for char in str1:
        char = char.lower()
        if char in countMap:
            countMap[char] += 1
        else:
            countMap.update({char: 1})

    print(countMap)

    stocks = {
        'AAPL': 191.88,
        'GOOG': 1186.96,
        'IBM': 149.24,
        'ORCL': 48.44,
        'ACN': 166.89,
        'FB': 208.09,
        'SYMC': 21.29
    }
    # 过滤出值大于100的数据
    over100 = {key: value for key, value in stocks.items() if value > 100}
    print(over100)


    pass


if __name__ == '__main__':
    # print(p03_01(10))
    # p03_02()
    # p03_03()
    # p03_05()
    # p03_06()
    # p03_08()
    # p03_09()
    # p03_10()
    # p03_11()
    # p03_12()
    p03_13()

    pass
