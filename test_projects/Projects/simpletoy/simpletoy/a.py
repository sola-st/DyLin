def foo():
    l = [1, 2, 3, 4]
    for i in l:
        if i < 4:
            l.pop(l.index(i))
    return "foo"

def bar():
    res = foo() + " bar"
    return res
