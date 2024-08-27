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
        a = a + "x"  # No warning because of performance
        b = "x" + b  # No warning because of performance
        b += a  # DyLin warn
        b = a + b  # No warning because of performance
        a = b + a  # No warning because of performance

    '''
    fixed cases
    '''
    x = 1
    x += 1
    x = x + 1


test()
