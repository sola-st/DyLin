from simpletoy import a as simpleA

def test_foo():
    res = simpleA.foo()
    assert res == "foo"

def test_bar():
    res = simpleA.bar()
    assert res == "foo bar"