
# 默认情况下，import mypkg 只会加载 mypkg/__init__.py，
# 不会自动导入 mypkg.utils 里的内容。

# ✅ 解决方案一：从子模块显式导入
# 你可以直接导入定义函数的模块：
# from mypkg.utils import say_hello


# 🧭__init__.py 可以做什么
# ① 定义包被导入时要执行的代码
print("📦 p06_package1 已被导入")

# ② 控制包导出哪些模块（__all__）
# __all__ = ['module1', 'module2']

# ③ 导入常用模块或函数，简化使用路径
# from .module1 import say_hello
# from .module2 import say_goodbye
# 这样外部可以直接：
# from my_package import say_hello

# ④ 包级变量、版本号、配置初始化
__version__ = '1.0.0'
config = {"debug": True}