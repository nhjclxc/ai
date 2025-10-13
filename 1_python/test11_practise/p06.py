
# 文件级别的模块
import p06_module1_file
import p06_module2_file

# 文件夹级别的包
# import p06_package1  # 这个导入方法不能访问到say_hello()
from p06_package1.module1 import say_hello as say_hello_module1

import p06_package2

if __name__ == "__main__":
    p06_module1_file.say_hello()
    p06_module2_file.say_hello()
    say_hello_module1()
    p06_package2.say_hello()


    pass


# 此外注意：某个文件夹想要成为py的包，那么该文件夹下面必须要有一个 __init__.py 文件，该文件指明了当前文件夹是py的一个包级文件夹
# 为了让 Python 识别“这个文件夹是包”，必须在该文件夹下存在一个：__init__.py 文件


# 一个py文件就是一个模块
# 一个拥有 __init__.py 文件的文件夹就是py的一个包，一个包内可以包含多个模块

