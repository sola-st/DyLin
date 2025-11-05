a = list(range(0, 32131))
b = list(range(0, 32132))
c = list(range(231, 321033))
d = list(range(0, 32134))
e = list(range(0, 32135))
f = list(range(0, 32136))
h = list(range(0, 32137))

sorted(a)  # DyLin warn
x = sorted(b)  # DyLin warn
y = sorted(c)  # DyLin warn
h = sorted(h, reverse=True)  # DyLin warn

d.sort()
z = sorted(e)
e.append([])
k = sorted(f)
f
h

# TODO example which only works for dynamic analysis

# a,b,c should be flagged

# Examples to not warn about from PR #4
a_set = set(range(0, 32131))
sorted_set = sorted(a_set)
a_string = "test string" * 1000
sorted_string = sorted(a_string)
a_tuple = tuple(range(0, 32131))
sorted_tuple = sorted(a_tuple)
a_dict = {i: i for i in range(0, 32131)}
sorted_dict = sorted(a_dict)


# Examples to still warn about from PR #4
class MyList(list):
    pass


my_list = MyList(range(0, 32131))
sorted_my_list = sorted(my_list)  # DyLin warn
