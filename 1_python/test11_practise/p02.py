

"""
输入三个整数x,y,z，请把这三个数由大到小输出。
"""

def fun1():
    global x, y
    x = int(input("输入第1个数"))
    y = int(input("输入第2个数"))
    z = int(input("输入第3个数"))
    temp = 0
    if y > x:
        temp = x
        x = y
        y = temp
    if z > x:
        temp = x
        x = z
        z = temp
    print(f"由小到大输出：{x} > {y} > {z}")


def fun2():
    nums = []
    for i in range(3):
        num = int(input("请输入第" + str(i + 1) + "个数："))
        nums.append(num)
    nums.sort()
    nums.reverse()
    print(nums)


if __name__ == '__main__':
    # fun1()

    fun2()

    pass
