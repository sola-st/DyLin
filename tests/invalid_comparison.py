import numpy as np

'''
setup
'''
d = { "Invalid Comparison": "InvalidComparisonAnalysis"}

def test_cmp_func():
    def a():
        pass
    def b():
        pass

    f'START;'
    a == ""
    f'END; a == ""'
    f'START;'
    b == 2
    f'END; b == 2'
    f'START;'
    2 == b
    f'END; 2 == b'
    f'START;'
    [] == a
    f'END; [] == a'

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
    f'START;'
    type(0) == type("")
    f'END; type(0) == type("")'
    f'START;'
    type(0) == type(0)
    f'END; type(0) == type(0)'
    f'START;'
    type("") == type(0)
    f'END; type("") == type(0)'
    f'START;'
    type("") == type("")
    f'END; type("") == type("")'
    f'START;'
    type(0) is type("")
    f'END; type(0) is type("")'
    f'START;'
    type(0) is type(0)
    f'END; type(0) is type(0)'

    2 == 3
    type(0) is 2
    "" == ""
    -2 is ""
    type(0) == 1
    1 == type(231)
    type("") is 2
    


def test_comparison():

    class SomeType():
        def __init__(self):
            pass
    class AnotherType():
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

    all_builtin_types = [type_string, type_integer, type_float, type_complex, type_list,
                        type_tuple, type_range, type_dict, type_set, type_frozenset, type_bool, type_bytes,
                        type_bytearray, type_memoryview]

    '''
    buggy cases
    '''
    for i in range(0,len(all_builtin_types)-1):
        # custom types
        f'START;'
        type_some_type == all_builtin_types[i]
        f'END; {type_some_type} == {all_builtin_types[i]}'
        f'START;'
        all_builtin_types[i] == type_some_type
        f'END; {all_builtin_types[i]} == {type_some_type}'
        for j in range(0,len(all_builtin_types)-1):
            if (i != j 
                and not isinstance(all_builtin_types[i], type(all_builtin_types[j])) 
                and not isinstance(all_builtin_types[j], type(all_builtin_types[i]))):

                # TODO don't compare int - float / float - int

                # builtins
                f'START;'
                all_builtin_types[i] == all_builtin_types[j]
                f'END; {all_builtin_types[i]} == {all_builtin_types[j]}"'

    # custom types, no inheritance
    f'START;'
    type_some_type == type_another_type
    f'END; {all_builtin_types[i]} == {all_builtin_types[j]}'
    f'START;'
    type_another_type == type_some_type
    f'END; {all_builtin_types[i]} == {all_builtin_types[j]}'

    '''
    fixed cases
    '''
    for i in range(0,len(all_builtin_types)):
        all_builtin_types[i] == all_builtin_types[i]

    # checks for inherited types where comparison is valid
    type_inerhited_some_type == type_some_type
    type_some_type == type_inerhited_some_type

def test_list_in_type_mismatch(asSet: bool):
    a = ["a"]
    b = ["a", "b", "c", "d"]
    c = [3]
    d = list(range(0,100))
    e = [["a"]]
    f = [["a"], ["b"], ["c"]]

    if asSet:
        a = set(a)
        b = set(b)
        c = set(c)
        d = set(d)
        # e,f are not hashable

    f'START;'
    a in b
    f'END;'
    f'START;'
    c in d
    f'END;'
    f'START;'
    e in f
    f'End;'

    b in c
    a in d
    f in e

def test_difference_is_eq_operators():
    # equal to False
    ef_1 = 0
    ef_2 = 0.0

    # equal to True
    et_1 = 1
    et_2 = 1.0

    f'START;'
    ef_1 == False
    f'END; ef_1 == False'
    f'START;'
    ef_2 != False
    f'END; ef_2 != False'

    f'START;'
    et_1 == True 
    f'END; ef_1 == True'
    f'START;'
    et_2 != True
    f'END; ef_2 != True'

def test_bad_floats():
    bad_float = 0.2 + 0.1
    bad_float_2 = 0.2 + 0.01

    f'START;'
    bad_float == 0.3
    f'END; bad_float == 0.05'
    f'START;'
    0.3 == bad_float
    f'END; 0.05 == bad_float'
    f'START;'
    bad_float != 0.3
    f'END; bad_float != 0.05'
    f'START;'
    0.3 != bad_float
    f'END; 0.05 != bad_float'

    f'START;'
    bad_float_2 == 0.21
    f'END; bad_float_2 == 0.21'
    f'START;'
    0.21 == bad_float_2
    f'END; 0.21 == bad_float_2'
    f'START;'
    bad_float_2 != 0.21
    f'END; bad_float_2 != 0.21'
    f'START;'
    0.21 != bad_float_2
    f'END; 0.21 != bad_float_2'

    bad_float == 0.03
    0.03 == bad_float
    bad_float_2 == bad_float
    bad_float_2 == 312321.0

def test_numpy():
    f'START;'
    np.float16("NaN") == 2
    f'END np.float16("NaN") == 2'
    f'START;'
    2 == np.float16("NaN")
    f'END 2 == np.float16("NaN")'

    f'START;'
    np.float32("NaN") == 2
    f'END np.float32("NaN") == 2'
    f'START;'
    2 == np.float32("NaN")
    f'END 2 == np.float32("NaN")'

    f'START;'
    np.float128("NaN") == 2
    f'END np.float128("NaN") == 2'
    f'START;'
    2 == np.float128("NaN")
    f'END 2 == np.float128("NaN")'

    f'START;'
    np.float16("inf") == 2
    f'END np.float16("inf") == 2'
    f'START;'
    2 == np.float16("inf")
    f'END 2 == np.float16("inf")'

    f'START;'
    np.float32("inf") == 2
    f'END np.float32("inf") == 2'
    f'START;'
    2 == np.float32("inf")
    f'END 2 == np.float32("inf")'

    f'START;'
    np.float128("inf") == 2
    f'END np.float128("inf") == 2'
    f'START;'
    2 == np.float128("inf")
    f'END 2 == np.float128("inf")'

    np.float16(2) == 2
    2 == np.float16(2)

    np.float32(2) == 2
    2 == np.float32(2)

    np.float128(2) == 2
    2 == np.float128(2)


test_cmp_func()
test_cmp_types()
test_bad_floats()
#test_comparison()
test_difference_is_eq_operators()
test_numpy()