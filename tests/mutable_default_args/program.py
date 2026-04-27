import uuid

# Fixture for functions that accidentally share mutable default arguments between calls.

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

    # Buggy cases: the default container is created once and then reused across calls.

    def a(x=[]):
        x.append("test")

    # The first call initializes shared state; the second call observes the unintended reuse.
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

    # Safe pattern: use None as a sentinel and allocate a fresh container per invocation.

    def x(x=None):
        if x is None:
            x = []
        x.append("test")

    x()
    x()


# Execute the fixture so the analysis sees both the buggy and corrected patterns.
test()
