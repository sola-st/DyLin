d = { "InPlace Sort": "InPlaceSortAnalysis"}

a = list(range(0,32132))
b = list(range(0,32132))
c = list(range(231,321032))
d = list(range(0,32132))
e = list(range(0,32132))
f = list(range(0,32132))

f'START;'
sorted(a)
f'START;'
x = sorted(b)
f'START;'
y = sorted(c)

d.sort()
z = sorted(e)
e.append([])
k = sorted(f)
f

# TODO example which only works for dynamic analysis

# a,b,c should be flagged
f'END;'
f'END;'
f'END;'