import math
import random
import secrets

DIGITS = '0123456789'
LETTERS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
DIGIT_CHARACTERS = DIGITS + LETTERS


def generate_code1(gen_type=3, code_len=4):
    chars = DIGIT_CHARACTERS
    if gen_type == 1:
        chars = DIGITS
    elif gen_type == 2:
        chars = LETTERS

    code = ''
    for i in range(code_len):
        code += chars[random.randrange(len(chars))]

    return code


# ✅ 方法 2：用 random.choice() 简化循环
# 更“Pythonic” 的写法：
def generate_code2(gen_type=3, code_len=4):
    chars = DIGIT_CHARACTERS
    if gen_type == 1:
        chars = DIGITS
    elif gen_type == 2:
        chars = LETTERS
    print("11111", random.sample(chars, k=code_len))  # sample属于无放回取样，不会重复
    print("22222", random.choices(chars, k=code_len))  # choices属于有放回取样，可能会重复
    # 随机选择 code_len 个字符
    return ''.join(random.choice(chars) for _ in range(code_len))


# ✅ 方法 3：用 secrets（更安全，推荐生成验证码）
# 如果你的验证码用于登录注册场景，推荐使用 Python 的 secrets 模块，更安全（适用于密码/验证码生成）：
def generate_code3(gen_type=3, code_len=4):
    chars = DIGIT_CHARACTERS
    if gen_type == 1:
        chars = DIGITS
    elif gen_type == 2:
        chars = LETTERS

    # 随机选择 code_len 个字符
    return ''.join(secrets.choice(chars) for _ in range(code_len))


def test1():
    print(generate_code1(1, 5))
    print(generate_code1(2, 5))
    print(generate_code1(3, 5))
    print(generate_code2(1, 5))
    print(generate_code2(2, 5))
    print(generate_code2(3, 5))
    print(generate_code3(1, 5))
    print(generate_code3(2, 5))
    print(generate_code3(3, 5))


# def mean(data):
#     return sum(data) / len(data)
#
#
# def median(data):
#     return sorted(data)[len(data) // 2]
#
#
# def var(data):
#     mean_val = mean(data)
#     return sum([pow(x - mean_val, 2)for x in data]) / (len(data) - 1)
#
#
# def std(data):
#     return pow(var(data), 0.5)
#
#
# def cv(data):
#     return std(data) / mean(data)
#
#
# def describe():
#     """输出描述性统计信息"""
#     data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#     print(f'均值: {mean(data)}')
#     print(f'中位数: {median(data)}')
#     print(f'方差: {var(data)}')
#     print(f'标准差: {std(data)}')
#     print(f'变异系数: {cv(data)}')

def ptp(data):
    """极差（全距）"""
    return max(data) - min(data)


def mean(data):
    """算术平均"""
    return sum(data) / len(data)


def median(data):
    """中位数"""
    temp, size = sorted(data), len(data)
    if size % 2 != 0:
        return temp[size // 2]
    else:
        return mean(temp[size // 2 - 1:size // 2 + 1])


def var(data, ddof=1):
    """方差"""
    x_bar = mean(data)
    temp = [(num - x_bar) ** 2 for num in data]
    return sum(temp) / (len(temp) - ddof)


def std(data, ddof=1):
    """标准差"""
    return var(data, ddof) ** 0.5


def cv(data, ddof=1):
    """变异系数"""
    return std(data, ddof) / mean(data)


def describe():
    """输出描述性统计信息"""
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    print(f'均值: {mean(data)}')
    print(f'中位数: {median(data)}')
    print(f'极差: {ptp(data)}')
    print(f'方差: {var(data)}')
    print(f'标准差: {std(data)}')
    print(f'变异系数: {cv(data)}')


def calc(data):
    total = 0
    for item in data:
        if type(item) in (int, float):
            total += item
    return total


def test3():
    """学会判断函数参数类型进行不同操作"""


    data = [1, 2, 3, 'x', 5.1, 6.2, 'zxc', 8, 9, 'awsd']

    # 计算int类型和folat类型的数据总和
    print(f"总和 = {calc(data)}")


    pass


def test5():
    """
        lambda函数
        定义：lambda 参数列表 : 表达式
    """

    add = lambda x, y: x+y

    print(add(1,2))


if __name__ == '__main__':
    # test1()

    # describe()

    # test3()

    # test5()

    test6()



    pass
