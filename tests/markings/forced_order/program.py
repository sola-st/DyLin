from typing import Set

d = {"Forced Order Test": "ObjectMarkingAnalysis", "configName": "forced_order"}

a = 1
b = 2
c, d = a, b
my_set = set()  # source: adds "unordered"
my_set.add("hey")
my_set.add("ho")
tmp = my_set
s = str(tmp)  # source: adds "forcedOrder"
# s = ''.join(sorted(s)) # source for sorted: removed unordered
s.startswith("{'hey")  # sink: throws e iff unordered and forcedOrder # DyLin warn

s = set()
s.add("a")
s.add("b")
res = ''.join(s)
res.startswith(res)  # DyLin warn
res.endswith(res)  # DyLin warn
res.split(",")  # DyLin warn
res.index("a")  # DyLin warn

s2 = set()
s2.add("a")
s2.add("b")
_list = list(s2)
_list.reverse()  # DyLin warn
_list.index("b")  # DyLin warn
_list.pop()  # DyLin warn

s.add("a")
s.add("b")
tup = tuple(s)
tup.index("b")  # DyLin warn

my_set = set()  # source: adds "unordered"
my_set.add("hey")
my_set.add("ho")
tmp = my_set
s = str(tmp)  # source: adds "forcedOrder"
s = ''.join(sorted(s))  # source for sorted: removed unordered
s.startswith("{'hey")  # sink: throws e iff unordered and forcedOrder


class A:
    def __init__(self):
        self.x = set()

    def a(self):
        self.x.add("hey")
        self.x.add("ho")

    def b(self):
        s = str(self.x)
        s.startswith("{'hey")  # DyLin warn


a = A()
a.a()
a.b()


class B:
    def a(self) -> Set[str]:
        x = set()
        x.add("hey")
        x.add("ho")
        return x

    def b(self):
        x = self.a()
        s = str(x)
        s.startswith("{'hey")  # DyLin warn


b = B()
b.b()


class C:
    def a(self):
        x = B().a()
        s = str(x)
        s.startswith("{'hey")  # DyLin warn


C().a()
