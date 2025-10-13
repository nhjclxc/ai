import itertools



def test01():
    # 题目：有四个数字：1、2、3、4，能组成多少个互不相同且无重复数字的三位数？各是多少？
    # 程序分析：可填在百位、十位、个位的数字都是1、2、3、4。组成所有的排列后再去 掉不满足条件的排列

    count = 0
    for i in range(1, 5):
        for j in range(1, 5):
            for k in range(1, 5):
                if i != j and i != k and j != k:
                    count = count + 1
                    print(i, j, k)

    print("count = ", count)
    pass


def test02():
    count = 0
    nums = [1, 2, 3, 4]
    for i in nums:  # 百位
        for j in nums:  # 十位
            for k in nums:  # 个位
                if i != j and i != k and j != k:
                    print(f"{i}{j}{k}")
                    count += 1

    print("总数为：", count)


def test03():

    nums = [1, 2, 3, 4]
    perms = list(itertools.permutations(nums, 3))
    for p in perms:
        print("".join(map(str, p)))
    print("总数为：", len(perms))

if __name__ == '__main__':
    test01()
    test02()
    test03()
    pass
