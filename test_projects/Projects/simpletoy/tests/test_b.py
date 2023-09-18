from simpletoy import b as simpleB

def test_baz():
    res = simpleB.baz(" ")
    assert res == "foo foo"