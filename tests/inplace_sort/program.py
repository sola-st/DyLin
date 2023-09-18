a = list(range(0, 32132))
b = list(range(0, 32132))
c = list(range(231, 321032))
d = list(range(0, 32132))
e = list(range(0, 32132))
f = list(range(0, 32132))

sorted(a)  # DyLin warn
x = sorted(b)  # DyLin warn
y = sorted(c)  # DyLin warn

d.sort()
z = sorted(e)
e.append([])
k = sorted(f)
f

# TODO example which only works for dynamic analysis

# a,b,c should be flagged
