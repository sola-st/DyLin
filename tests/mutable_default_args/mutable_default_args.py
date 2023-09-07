import uuid 

def test():
    d = { "Mutable Default Args": "MutableDefaultArgsAnalysis"}

    # TODO try python classes as mutable types
    class SomeType():
        def __init__(self):
            pass
    class AnotherType():
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
    f'START;'
    a()
    f'END; list as default value used'

    def b(x2={}):
        x2[str(uuid.uuid4())] = "x"
    b()
    f'START;'
    b()
    f'END; dict as default value used'

    def c(x=set()):
        x.add("test")
    c()
    f'START;'
    c()
    f'END; set as default value used'

    
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