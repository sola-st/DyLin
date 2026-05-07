def test():
    '''
    buggy cases
    '''
    a = "a"
    b = "b"

    for i in range(0, 10001):
        # to prevent actual memory issues
        a = "a"
        b = "b"

        a += b  # DyLin warn
        a += ""  # DyLin warn
        a = a + "x"  # DyLin warn
        b = "x" + b  # DyLin warn
        b += a  # DyLin warn
        b = a + b  # DyLin warn
        a = b + a  # DyLin warn

    '''
    fixed cases
    '''
    x = 1
    x += 1
    x = x + 1


test()
