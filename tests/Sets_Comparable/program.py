string_set = {"banana", "apple", "cherry"}
sorted_set = sorted(string_set) # OK


mixed_set = {3, "banana", 1, "apple"}
sorted_set = sorted(mixed_set, key=str) # OK


string_set = {"banana", "apple", "cherry"}
sorted_set = sorted(string_set, key=len) # OK


mixed_set = {3, "banana", 1, "apple"}

try:
    # This will raise a TypeError because integers and strings cannot be compared
    sorted_set = sorted(mixed_set) # DyLin warn
except TypeError as e:
    pass
