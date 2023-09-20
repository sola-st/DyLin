import numpy as np


def test_cmp_func():
    def a():
        pass

    def b():
        pass

    a == ""  # DyLin warn
    b == 2  # DyLin warn
    2 == b  # DyLin warn
    [] == a  # DyLin warn

    a == int.__abs__
    int.__abs__ == a
    a == b
    b == a
    a() == b()
    a() == 2
    b() == ""
    2 == b()
    "" == a()


def test_cmp_types():
    type(0) == type("")  # DyLin warn
    type(0) == type(0)
    type("") == type(0)  # DyLin warn
    type("") == type("")
    type(0) is type("")
    type(0) is type(0)

    2 == 3
    two = 2
    type(0) is two
    "" == ""
    minus_two = -2
    empty_string = ""
    minus_two is empty_string
    type(0) == 1
    1 == type(231)
    type("") is two


def test_comparison():
    class SomeType:
        def __init__(self):
            pass

    class AnotherType:
        def __init__(self):
            pass

    class InerhitedType(SomeType):
        pass

    # python builtins
    type_string = str("Hello World")
    type_integer = int(20)
    type_float = float(20.5)
    type_complex = complex(1j)
    type_list = list(("apple", "banana", "cherry"))
    type_tuple = tuple(("apple", "banana", "cherry"))
    type_range = range(6)
    type_dict = dict(name="John", age=36)
    type_set = set(("apple", "banana", "cherry"))
    type_frozenset = frozenset(("apple", "banana", "cherry"))
    type_bool = bool(5)
    type_bytes = bytes(5)
    type_bytearray = bytearray(5)
    type_memoryview = memoryview(bytes(5))

    # created types
    type_some_type = SomeType()
    type_inerhited_some_type = InerhitedType()
    type_another_type = AnotherType()

    all_builtin_types = [
        type_string,
        type_integer,
        type_float,
        type_complex,
        type_list,
        type_tuple,
        type_range,
        type_dict,
        type_set,
        type_frozenset,
        type_bool,
        type_bytes,
        type_bytearray,
        type_memoryview,
    ]

    '''
    buggy cases
    '''
    for i in range(len(all_builtin_types)):
        # custom types
        type(type_some_type) == type(all_builtin_types[i])  # DyLin warn
        type(all_builtin_types[i]) == type(type_some_type)  # DyLin warn
        for j in range(len(all_builtin_types)):
            if (
                i != j
                and not isinstance(all_builtin_types[i], type(all_builtin_types[j]))
                and not isinstance(all_builtin_types[j], type(all_builtin_types[i]))
            ):
                # TODO don't compare int - float / float - int

                # builtins
                type(all_builtin_types[i]) == type(all_builtin_types[j])  # DyLin warn

    # custom types, no inheritance
    type(type_some_type) == type(type_another_type)  # DyLin warn
    type(type_another_type) == type(type_some_type)  # DyLin warn

    # inherited types
    type(type_inerhited_some_type) == type(type_some_type)  # DyLin warn
    type(type_some_type) == type(type_inerhited_some_type)  # DyLin warn

    '''
    fixed cases
    '''
    for i in range(len(all_builtin_types)):
        type(all_builtin_types[i]) == type(all_builtin_types[i])


def test_list_in_type_mismatch(asSet: bool):
    a = ["a"]
    b = ["a", "b", "c", "d"]
    c = [3]
    d = list(range(0, 100))
    e = [["a"]]
    f = [["a"], ["b"], ["c"]]

    if asSet:
        a = set(a)
        b = set(b)
        c = set(c)
        d = set(d)
        # e,f are not hashable

    a in b  # DyLin warn
    c in d  # DyLin warn
    e in f  # DyLin warn

    b in c
    a in d
    f in e
    a in f


def test_difference_is_eq_operators():
    # equal to False
    ef_1 = 0
    ef_2 = 0.0

    # equal to True
    et_1 = 1
    et_2 = 1.0

    ef_1 == False  # DyLin warn
    ef_2 != False  # DyLin warn

    et_1 == True  # DyLin warn
    et_2 != True  # DyLin warn


def test_bad_floats():
    bad_float = 0.2 + 0.1
    bad_float_2 = 0.2 + 0.01

    bad_float == 0.3  # DyLin warn
    0.3 == bad_float  # DyLin warn
    bad_float != 0.3  # DyLin warn
    0.3 != bad_float  # DyLin warn

    bad_float_2 == 0.21  # DyLin warn
    0.21 == bad_float_2  # DyLin warn
    bad_float_2 != 0.21  # DyLin warn
    0.21 != bad_float_2  # DyLin warn

    bad_float == 0.03
    0.03 == bad_float
    bad_float_2 == bad_float
    bad_float_2 == 312321.0


def test_numpy():
    np.float16("NaN") == 2  # DyLin warn
    2 == np.float16("NaN")  # DyLin warn

    np.float32("NaN") == 2  # DyLin warn
    2 == np.float32("NaN")  # DyLin warn

    np.float128("NaN") == 2  # DyLin warn
    2 == np.float128("NaN")  # DyLin warn

    np.float16("inf") == 2  # DyLin warn
    2 == np.float16("inf")  # DyLin warn

    np.float32("inf") == 2  # DyLin warn
    2 == np.float32("inf")  # DyLin warn

    np.float128("inf") == 2  # DyLin warn
    2 == np.float128("inf")  # DyLin warn

    np.float16(2) == 2
    2 == np.float16(2)

    np.float32(2) == 2
    2 == np.float32(2)

    np.float128(2) == 2
    2 == np.float128(2)


test_cmp_func()
test_cmp_types()
test_bad_floats()
test_comparison()
test_difference_is_eq_operators()
test_numpy()
