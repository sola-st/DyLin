string_list = ["banana", "apple", "cherry"]
sorted_list = sorted(string_list) # OK


mixed_list = [3, "banana", 1, "apple"]
sorted_list = sorted(mixed_list, key=str) # OK


string_list = ["banana", "apple", "cherry"]
sorted_list = sorted(string_list, key=len) # OK


mixed_list = [3, "banana", 1, "apple"]

try:
    # This will raise a TypeError because integers and strings cannot be compared
    sorted_list = sorted(mixed_list) # DyLin warn
except TypeError as e:
    pass