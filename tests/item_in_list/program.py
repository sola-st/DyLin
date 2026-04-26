# Fixture for membership checks whose runtime cost depends heavily on container choice.
# Large lists require a linear scan for "in", which is what this analysis flags.
l = list(range(10000))
if 123 in l:  # DyLin warn
    print("Found")

# Iteration itself is fine here; the warning is specifically about repeated membership search.
for i in l:
    pass

# String membership is a normal control case and should not trigger this rule.
if "a" in "hello world":
    print("Found")
