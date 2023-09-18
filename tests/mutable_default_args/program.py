import uuid


def test():
    # TODO try python classes as mutable types
    class SomeType:
        def __init__(self):
            pass

    class AnotherType:
        def __init__(self):
            pass

    class InerhitedType(SomeType):
        pass

    '''
    buggy cases
    '''

    def a(x=[]):
        x.append("test")

    a()
    a()  # DyLin warn

    def b(x2={}):
        x2[str(uuid.uuid4())] = "x"

    b()
    b()  # DyLin warn

    def c(x=set()):
        x.add("test")

    c()
    c()  # DyLin warn

    '''
    fixed cases
    '''

    def x(x=None):
        if x is None:
            x = []
        x.append("test")

    x()
    x()


test()
