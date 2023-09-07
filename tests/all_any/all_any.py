d = { "All Any Analysis": "BuiltinAllAnalysis"}

all([True, True, True])
all([True, True, False])

f'START;'
all([]) #returns True
f'END; all([])'

all([[]]) # returns False

f'START;'
all([[[]]]) # returns True
f'END; all([[[]]])'

f'START;'
all([[[[]]]]) # returns True
f'END; all([[[[]]]])'

all([[[True]]])
all([[[], True]])
all([[[]], True])

any([[[True]]])
any([[[], True]])
any([[[]], True])

f'START;'
any([[[]]])
f'END; any([[[]]])'