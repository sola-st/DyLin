l = list(range(10000))
if 123 in l:  # DyLin warn
    print("Found")

for i in l:
    pass

if "a" in "hello world":
    print("Found")
