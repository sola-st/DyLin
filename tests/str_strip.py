d = { "String Strip Test": "StringStripAnalysis"}

f'START;'
"".strip("abab")
f'END;'
a = "xyzz"
f'START;'
"".strip(a)
f'END;'
f'START;'
"1,2".strip(','.join([str(s) for s in range(0,999)]))
f'END;'

"".strip("a")
"".strip("abc")
"".strip(''.join([str(s) for s in range(0,9)]))