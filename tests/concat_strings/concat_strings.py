def check(i, is_start, msg = ""):
    if i == 10000:
        if is_start:
            f'START;'
        else:
            f'END;{msg}'

def test():
    d = { "String Concat": "StringConcatAnalysis"}

    '''
    buggy cases
    '''
    a = "a"
    b = "b"

    for i in range(0,10001):
        # to prevent actual memory issues
        a = "a"
        b = "b"

        check(i, True)
        a += b
        check(i, False, 'a+=b')
        check(i, True)
        a += ""
        check(i, False, 'a += ""')
        check(i, True)
        a = a + "x"
        check(i, False, 'a = a + "x"')
        check(i, True)
        b = "x" + b
        check(i, False, 'b = "x" + b')
        check(i, True)
        b += a
        check(i, False, 'b += a')
        check(i, True)
        b = a + b
        check(i, False, 'b = a + b')
        check(i, True)
        a = b + a
        check(i, False, 'a = b + a')

    '''
    fixed cases
    '''
    x = 1
    x += 1
    x = x + 1

test()