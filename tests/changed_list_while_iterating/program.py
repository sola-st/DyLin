def get_list():
    return [1, 2, 3, 4]


list_1 = get_list()
for item in list_1:
    del item
# resulting list is [1,2,3,4], should not warn

list_2 = get_list()
for item in list_2:  # DyLin warn
    list_2.remove(item)
# resulting list is [2,4], should warn

list_3 = get_list()
for item in list_3[:]:
    list_3.remove(item)
# resulting list is [], should not warn, object is copied

list_4 = get_list()
list_x = list_4
for item in list_x:  # DyLin warn
    list_4.remove(item)
# resulting list is [2,4], should warn

for item in list_x:
    list_x[0] = "test" * 1000000
# should not warn, does not change list size

list_6 = get_list()
for item in list_6:  # DyLin warn
    list_6.pop()
# resulting list is [1,2], should warn


def f(x):
    x.pop()


list_7 = get_list()
for item in list_7:  # DyLin warn
    f(list_7)
# same as list_6, should warn

list_9 = get_list()
for idx, item in enumerate(list_9):
    list_9.pop(idx)
# uses generator, not supported (yet)

a = ", ".join(str(choice) for choice in get_list())
