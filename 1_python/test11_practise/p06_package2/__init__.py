
# ✅ 解决方案二：在 __init__.py 中手动暴露方法

from .module2 import say_hello

# 使用 导出 相关方法和变量
__all__ = ["say_hello"]

