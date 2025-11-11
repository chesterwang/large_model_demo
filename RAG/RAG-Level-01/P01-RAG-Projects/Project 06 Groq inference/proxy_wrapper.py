from contextlib import contextmanager
import os


# 定义一个上下文
@contextmanager
def my_context():
    print("[enter context]")
    yield
    print("[exit context]")

@contextmanager
def GFW_proxy():
    os.environ['HTTP_PROXY'] = "http://127.0.0.1:7890"
    os.environ['HTTPS_PROXY'] = "http://127.0.0.1:7890"
    yield
    del os.environ['HTTP_PROXY']
    del os.environ['HTTPS_PROXY']

def wrap_method_with_context(instance, method_name, context_manager):
    """将实例的某个方法包装，使其执行时自动进入context"""
    orig_method = getattr(instance, method_name)

    def wrapped(*args, **kwargs):
        with context_manager():
            return orig_method(*args, **kwargs)

    setattr(instance, method_name, wrapped)

## ---

def test1():
    class MyClass:
        def foo(self):
            print("running foo")

        def bar(self):
            print("running bar")

    # ✅ 示例使用
    obj = MyClass()
    wrap_method_with_context(obj, "foo", my_context)

    obj.foo()  # 会自动进入上下文
    obj.bar()  # 不进入上下文

def test2():
    class CCC:
        def AAA(self):
            pass

    class MyClass(CCC):
        def foo(self):
            print("running foo")
            print(os.environ.get('HTTP_PROXY',default=None))

        def bar(self):
            print("running bar")
            print(os.environ.get('HTTP_PROXY',default=None))

    # ✅ 示例使用
    obj = MyClass()
    wrap_method_with_context(obj, "AAA", GFW_proxy)

    # obj.foo()  # 会自动进入上下文
    # obj.bar()  # 不进入上下文
    obj.AAA()


    print(os.environ.get('HTTP_PROXY', default=None))

if __name__ == '__main__':
    test2()

