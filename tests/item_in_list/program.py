l = list(range(500))
if 23 in l:
    print("Found")

for i in range(30):
    if i in l:  # DyLin warn
        print("Found")

for i in l:
    pass

if "a" in "hello world":
    print("Found")
