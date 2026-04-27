import random

# Fixture for randrange() calls that incorrectly use keyword arguments.
# Control case: randrange() is meant to be called with positional bounds.
random.randrange(10) # OK

# Warning case: passing step as a keyword is unsupported and should be flagged.
random.randrange(10, 100, step=2) # DyLin warn
