# Filename: func2_returns.py
import time

from datetime import datetime
from arrow import now


def fun1():
    ''' 测试函数的多个返回值 '''
    now1 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(now1)
    print(datetime.now())  # 2025-10-09 11:25:33.627361
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))  # 2025-10-09 11:25:33
    print(datetime.now().strftime("%y-%m-%d %H:%M:%S"))  # 2025-10-09 11:25:33
    """
    | 格式符         | 含义               | 示例        |
    | ----------- | ---------------- | --------- |
    | `%Y`        | 年（四位）            | `2025`    |
    | `%y`        | 年（两位）            | `25`      |
    | `%m`        | 月（两位，01–12）      | `10`      |
    | `%B`        | 月份全名（英文）         | `October` |
    | `%b` / `%h` | 月份缩写（英文）         | `Oct`     |
    | `%d`        | 日期（两位，01–31）     | `09`      |
    | `%j`        | 一年中的第几天（001–366） | `282`     |
    | `%U`        | 一年中的第几周（周日为第一天）  | `40`      |
    | `%W`        | 一年中的第几周（周一为第一天）  | `40`      |
    
    | 格式符  | 含义                  | 示例           |
    | ---- | ------------------- | ------------ |
    | `%H` | 小时（24小时制，00–23）     | `14`         |
    | `%I` | 小时（12小时制，01–12）     | `02`         |
    | `%p` | 上午/下午（AM / PM）      | `PM`         |
    | `%M` | 分钟（00–59）           | `35`         |
    | `%S` | 秒（00–59）            | `08`         |
    | `%f` | 微秒（000000–999999）   | `123456`     |
    | `%z` | 时区偏移（+HHMM / -HHMM） | `+0800`      |
    | `%Z` | 时区名                 | `CST`, `UTC` |

    """

    # 由于tuple元组()的不可变性，所以多个函数返回值一般是使用元组来封装返回
    return 111, 'qwe', now()
    # return (111, 'qwe', now())


if __name__ == '__main__':
    tuple = fun1()
    for t in tuple:
        print(t)
