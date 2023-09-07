'''
setup
'''
def test(length=1001):
    d = {"Wrong type added": "WrongTypeAddedAnalysis"}

    l = [10] * length
    f'START;'
    l.append("test")
    f'END; add() wrong type "test"'
    l = [10] * length
    f'START;'
    l.extend(["testi", 10])
    f'END; extend() wrong type ["testi", 10]'
    l = [10] * length
    f'START;'
    l.insert(20,"test")
    f'END; insert() wrong type "test"'
    l = [10] * length
    f'START;'
    l += ["test"]
    f'END; += wrong type "test"'
    l = [10] * length
    f'START;'
    l = l + ["test"]
    f'END; + wrong type "test"'

    l = set(range(0,length))
    f'START;'
    l.add("test")
    f'END; add wrong type "test" to set'

    l = [10] * length
    l.append(10)
    l.extend([10])
    l.insert(20,10)
    l += [10]
    l = l + [10]

    l = set(range(0,length))
    l.add(32131313)

test()