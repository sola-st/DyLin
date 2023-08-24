from .a import foo

def baz(s: str):
    res = foo() + s + foo()
    return res