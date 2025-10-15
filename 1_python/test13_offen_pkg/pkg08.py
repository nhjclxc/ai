import operator

"""
| 函数                       | 功能       | 示例                   |
| ------------------------ | -------- | -------------------- |
| `operator.add(a,b)`      | `a + b`  | `add(2,3)` → 5       |
| `operator.sub(a,b)`      | `a - b`  | `sub(5,2)` → 3       |
| `operator.mul(a,b)`      | `a * b`  | `mul(2,3)` → 6       |
| `operator.truediv(a,b)`  | `a / b`  | `truediv(6,3)` → 2.0 |
| `operator.floordiv(a,b)` | `a // b` | `floordiv(7,3)` → 2  |
| `operator.mod(a,b)`      | `a % b`  | `mod(7,3)` → 1       |
| `operator.pow(a,b)`      | `a ** b` | `pow(2,3)` → 8       |
| `operator.neg(a)`        | `-a`     | `neg(5)` → -5        |
| `operator.pos(a)`        | `+a`     | `pos(-5)` → -5       |
| `operator.abs(a)`        | `abs(a)` | `abs(-7)` → 7        |


| 函数                 | 功能       | 示例               |
| ------------------ | -------- | ---------------- |
| `operator.eq(a,b)` | `a == b` | `eq(2,2)` → True |
| `operator.ne(a,b)` | `a != b` | `ne(2,3)` → True |
| `operator.gt(a,b)` | `a > b`  | `gt(5,3)` → True |
| `operator.ge(a,b)` | `a >= b` | `ge(5,5)` → True |
| `operator.lt(a,b)` | `a < b`  | `lt(2,3)` → True |
| `operator.le(a,b)` | `a <= b` | `le(2,3)` → True |

| 函数                   | 功能      | 示例                   |                |
| -------------------- | ------- | -------------------- | -------------- |
| `operator.and_(a,b)` | `a & b` | `and_(5,3)` → 1      |                |
| `operator.or_(a,b)`  | `a      | b`                   | `or_(5,3)` → 7 |
| `operator.xor(a,b)`  | `a ^ b` | `xor(5,3)` → 6       |                |
| `operator.not_(a)`   | `not a` | `not_(True)` → False |                |


"""

if __name__ == '__main__':
    # operator --- 内置操作符接口
    # Python 的 operator 模块提供了对 内置运算符和函数的函数化接口，方便在函数式编程、排序、map、reduce 等场景中使用。

    print(1 ^ 2)
    print(2 ^ 2)
    print(2 ^ 3)

    print("Hello World!")

    pass
