"".strip("abab")  # DyLin warn
a = "xyzz"
"".strip(a)  # DyLin warn
"1,2".strip(','.join([str(s) for s in range(0, 999)]))  # DyLin warn

"".strip("a")
"".strip("abc")
"".strip(''.join([str(s) for s in range(0, 9)]))
"foo.bar.rar".strip(".rar")  # DyLin warn
"foo.kab.bak".strip(".bak")  # DyLin warn
"<|en|>".strip("<|>")
