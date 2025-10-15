import difflib

"""
| 功能        | 方法                                           | 示例         |
| --------- | -------------------------------------------- | ---------- |
| 序列/字符串相似度 | `SequenceMatcher(None, a, b).ratio()`        | 0.8        |
| 匹配块          | `SequenceMatcher.get_matching_blocks()`      | 列表 Match   |
| 逐行/逐字符差异  | `difflib.ndiff(a,b)`                         | 带 + - 符号   |
| 统一差异格式    | `difflib.unified_diff(a,b)`                  | git风格 diff |
| 模糊匹配列表    | `difflib.get_close_matches(word, word_list)` | 返回最相似单词列表  |

"""

def test1():
    # 1️⃣ SequenceMatcher —— 字符/序列相似度
    # SequenceMatcher 可以比较两个字符串或序列，返回 相似度比例 或 差异信息

    # 相似度比例
    print(difflib.SequenceMatcher(None, "apple", "appl").ratio())
    print(difflib.SequenceMatcher(None, "apple", "app").ratio())
    print(difflib.SequenceMatcher(None, "apple", "ap").ratio())
    print(difflib.SequenceMatcher(None, "apple", "a").ratio())
    print(difflib.SequenceMatcher(None, "apple", "").ratio())
    # 0.8 （0~1，越接近1表示越相似）


def test2():

    # difflib.ndiff —— 行/字符差异
    # ndiff 可以显示两个序列的 逐字符差异（带 + - 符号）

    s1 = "bananandiff"
    s2 = "bananasndiff"

    s1 = "apple\nbanana\norange"
    s2 = "apple\nbananas\norange"

    diff = difflib.ndiff(s1.splitlines(), s2.splitlines())
    print('\n'.join(diff))

    pass



if __name__ == '__main__':
    """
    difflib -- 字符比较
    Python 的 difflib 模块是标准库中用于 序列比较、字符串差异比较和相似度计算 的工具，非常适合做文本比对、文件比对、模糊匹配等。
    
    import difflib

    """

    test1()
    test2()

    print("Hello World!")
