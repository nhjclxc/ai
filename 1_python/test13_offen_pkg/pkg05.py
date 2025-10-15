import collections


def test1():
    # 定义命名元组
    Point = collections.namedtuple('Point', ['x', 'y', 'z'])
    p = Point(10, 20, 30)

    print(p.x)  # 10
    print(p.y)  # 20
    print(p.z)  # 30
    print(p[0], p[1], p[2])  # 10，像普通元组一样支持索引

    # 常用方法：_fields、_replace、_asdict()
    print(p._fields)
    print(p._asdict())
    print(p._replace(x=100))

    pass


def test2():
    # 🧩 2. deque —— 双端队列
    # 可以 高效地在两端插入和删除，比 list 更快。

    dq = collections.deque([1, 2, 3, 4, 5, 6])
    dq.appendleft(8)
    dq.append(9)
    dq.append(9)
    dq.append(9)
    dq.extend([1,2,3])

    print(dq)

    print(dq.popleft())
    print(dq)

    dq2 = dq.copy()
    print(dq2)

    dq.append(999)
    print(dq)
    print(dq2)

    # 统计5这个数据出现了多少次
    print(dq.count(5))
    print(dq.count(9))
    # print(dq.count('"aaa"'))



    pass


def test3():
    # 🧩 3. Counter —— 计数器
    # 统计序列中元素出现次数，返回 字典-like 对象。

    c = collections.Counter(['a', 'a', 'b', 'c', 'a', 'b'])

    print(c)

    print(c.items())

    print(c.get('a'))
    print(c.get('c'))
    print(c.get('ccccc'))

    print(c.popitem())
    print(c.popitem())
    # print(c.pop('a'))
    print(c.most_common(2))

    pass


if __name__ == '__main__':
    """
    collections --- 数据类型容器
    
| 容器类型          | 主要用途     | 常用方法                                                         |
| ------------- | -------- | ------------------------------------------------------------ |
| `namedtuple`  | 可读的元组    | `_fields`, `_replace`, `_asdict()`                           |
| `deque`       | 双端队列     | `append`, `appendleft`, `pop`, `popleft`, `extend`, `rotate` |
| `Counter`     | 元素计数     | `most_common()`, `update()`, `elements()`                    |
| `OrderedDict` | 保持顺序的字典  | `move_to_end()`, `popitem()`, `keys()`                       |
| `defaultdict` | 自动初始化的字典 | 根据工厂函数生成默认值                                                  |
| `ChainMap`    | 合并多个字典   | `maps` 查看字典列表, 支持索引访问                                        |

    
    """

    # test1()

    # test2()

    test3()
