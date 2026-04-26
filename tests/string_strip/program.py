# Fixture for strip() calls that look like suffix removal but strip character sets instead.
# Repeated or dynamically built character sets can hide the fact that strip() works per character.
"".strip("abab")  # DyLin warn
a = "xyzz"
"".strip(a)  # DyLin warn
"1,2".strip(','.join([str(s) for s in range(0, 999)]))  # DyLin warn

# Small character sets are left here as control cases, even though strip() still treats them as sets.
"".strip("a")
"".strip("abc")
"".strip(''.join([str(s) for s in range(0, 9)]))

# These look like extension removal, but strip(".rar") and strip(".bak") remove any matching edge chars.
"foo.bar.rar".strip(".rar")  # DyLin warn
"foo.kab.bak".strip(".bak")  # DyLin warn

# This control case shows a deliberate character-set trim rather than a mistaken suffix removal.
"<|en|>".strip("<|>")
