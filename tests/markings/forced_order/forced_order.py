from typing import Set

d = { "Forced Order Test": "ObjectMarkingAnalysis", "configName": "forced_order"}

a = 1
b = 2
c,d = a,b
my_set = set() # source: adds "unordered"
my_set.add("hey")
my_set.add("ho")
tmp = my_set
s = str(tmp) # source: adds "forcedOrder"
#s = ''.join(sorted(s)) # source for sorted: removed unordered
f'START;'
s.startswith("{'hey") # sink: throws e iff unordered and forcedOrder
f'END; s.startwith has forced an order'

s = set()
s.add("a")
s.add("b")
res = ''.join(s)
f'START;'
res.startswith(res)
f'END; after "".join'
f'START;'
res.endswith(res)
f'END; after endswith'
f'START;'
res.split(",")
f'END; split'
f'START;'
res.index("a")
f'END; index'

s2 = set()
s2.add("a")
s2.add("b")
_list = list(s2)
f'START;'
_list.reverse()
f'END; list.reverse'
f'START;'
_list.index("b")
f'END; list.index'
f'START;'
_list.pop()
f'END; list.pop'

s.add("a")
s.add("b")
tup = tuple(s)
f'START;'
tup.index("b")
f'END; tuple.index'

my_set = set() # source: adds "unordered"
my_set.add("hey")
my_set.add("ho")
tmp = my_set
s = str(tmp) # source: adds "forcedOrder"
s = ''.join(sorted(s)) # source for sorted: removed unordered
s.startswith("{'hey") # sink: throws e iff unordered and forcedOrder

class A():
    def __init__(self):
        self.x = set()
    def a(self):
        self.x.add("hey")
        self.x.add("ho")

    def b(self):
        s = str(self.x)
        f'START;'
        s.startswith("{'hey")
        f'END; s.startswith in class'

a = A()
a.a()
a.b()

class B():
    def a(self) -> Set[str]:
        x = set()
        x.add("hey")
        x.add("ho")
        return x

    def b(self):
        x = self.a()
        s = str(x)
        f'START;'
        s.startswith("{'hey")
        f'END; s.startswith in after return'
b = B()
b.b()

class C():
    def a(self):
        x = B().a()
        s = str(x)
        f'START;'
        s.startswith("{'hey")
        f'END; s.startswith in after return from another class'

C().a()