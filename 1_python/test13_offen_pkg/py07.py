import itertools


def test1():
    """
    2️⃣ 无限迭代器（无限生成序列）
| 方法                           | 功能               | 示例                                         |
| ----------------------------- | ---------------- | ------------------------------------------ |
| `count(start=0, step=1)`     | 从 `start` 开始无限递增 | `for i in itertools.count(5, 2): print(i)` |
| `cycle(iterable)`            | 无限循环序列           | `for x in itertools.cycle('AB'):`          |
| `repeat(object, times=None)` | 重复某个元素           | `list(itertools.repeat(5, 3))` → `[5,5,5]` |

    """

    arr = itertools.count(start=2, step=1)
    print(arr.__next__())
    print(arr.__next__())
    print(arr.__next__())
    print(arr.__next__())
    print(arr.__next__())

    count = 0
    iterc = itertools.cycle(['a', 'b', 'c'])
    for i in iterc:
        print(i, end=', ')
        count += 1
        if count > 5:
            break

    print()

    aa = itertools.repeat('a', times=3)
    print(aa)
    print(list(aa))

    aaa = itertools.repeat([1, 2, 3], times=2)
    print(list(aaa))

    pass


def test2():
    """
3️⃣ 有限组合迭代器
| 方法                                           | 功能      | 示例                                                           |
| -------------------------------------------- | ------- | ------------------------------------------------------------ |
| `product(*iterables, repeat=1)`              | 笛卡尔积    | `itertools.product([1,2],[3,4])` → (1,3),(1,4),(2,3),(2,4)   |
| `permutations(iterable, r=None)`             | 排列      | `permutations([1,2,3],2)` → (1,2),(1,3),(2,1)...             |
| `combinations(iterable, r)`                  | 组合（不重复） | `combinations([1,2,3],2)` → (1,2),(1,3),(2,3)                |
| `combinations_with_replacement(iterable, r)` | 重复组合    | `combinations_with_replacement([1,2],2)` → (1,1),(1,2),(2,2) |


    """
    print(list(itertools.product([1,2,3], [8,9], repeat=1)))

    print(list(itertools.permutations([1,2,3])))
    print(list(itertools.permutations([1,2,3], 2)))
    print(list(itertools.permutations([1,2,3], 1)))

    print(list(itertools.combinations([1,2,3], 3)))
    print(list(itertools.combinations([1,2,3], 2)))
    print(list(itertools.combinations([1,2,3], 1)))



    pass


if __name__ == '__main__':
    """
    itertools --- 迭代器函数
        Python 的 itertools 是标准库中非常强大的 迭代器函数工具箱，适合生成器、无限序列、组合排列等操作。
    """

    # test1()

    test2()



    pass
