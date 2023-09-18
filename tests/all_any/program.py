all([True, True, True])
all([True, True, False])

all([])  # returns True # DyLin warn

all([[]])  # returns False

all([[[]]])  # returns True # DyLin warn

all([[[[]]]])  # returns True # DyLin warn

all([[[True]]])
all([[[], True]])
all([[[]], True])

any([[[True]]])
any([[[], True]])
any([[[]], True])

any([[[]]])  # DyLin warn
