import tempfile
from pathlib import Path

# Fixture for file handles that are read from but never explicitly closed.

def test():
    # Create a temporary directory so the fixture can open disposable files safely.
    tmpDir = Path(tempfile.mkdtemp())

    # Buggy case: the handle is left open after use, so the analysis should report it.
    buggy_file = open(tmpDir / '0', "a+")  # DyLin warn
    a = buggy_file.read()

    # Safe case: the context manager closes the file automatically on exit.
    with open(tmpDir / '1', "a+") as file1:
        pass

    # Safe case: explicit close() is also acceptable after the read completes.
    file2 = open(tmpDir / '2', "a+")
    a = file2.read()
    file2.close()


test()
