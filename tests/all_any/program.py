# Fixture for suspicious truthiness edge cases in all()/any() on nested empty containers.
# Simple baseline cases: these behave the way most readers expect.
all([True, True, True])
all([True, True, False])

# Empty iterables make all() vacuously True, which is easy to overlook.
all([])  # returns True # DyLin warn

# A directly nested empty list is still falsy, so this stays unflagged.
all([[]])  # returns False

# Extra nesting turns the outer element truthy again even though it still wraps emptiness.
all([[[]]])  # returns True # DyLin warn

all([[[[]]]])  # returns True # DyLin warn

# These nested mixtures are included as non-warning control cases.
all([[[True]]])
all([[[], True]])
all([[[]], True])

# any() only needs one truthy element, so these nested structures stay safe.
any([[[True]]])
any([[[], True]])
any([[[]], True])

# With only nested empty structure present, any() still sees no truthy element.
any([[[]]])  # DyLin warn
