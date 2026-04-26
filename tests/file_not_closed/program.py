import tempfile
from pathlib import Path


def test():
    '''
    setup
    '''
    tmpDir = Path(tempfile.mkdtemp())

    '''
    buggy cases
    '''
    buggy_file = open(tmpDir / '0', "a+")  # DyLin warn
    a = buggy_file.read()

    '''
    fixed cases
    '''
    with open(tmpDir / '1', "a+") as file1:
        pass

    file2 = open(tmpDir / '2', "a+")
    a = file2.read()
    file2.close()


test()
