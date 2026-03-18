"""
CLI demo project for DyLin.

Running this file with the DyLin CLI should produce findings for:
  - SL-01 InPlaceSort        (line 13) sorted() called on a list but result unused
  - PC-01 InvalidFunctionComparison (line 17) function compared to None with ==
"""

# SL-01: result of sorted() on a plain list is discarded should be flagged
nums = list(range(2000))
sorted(nums)  # DyLin: SL-01 expected here


# PC-01: comparing a function object to None with == should be flagged
def my_func():
    return 42

if my_func == 42:  # DyLin: PC-01 expected here  # noqa: E711
    print("my_func is 42")
  
result = my_func() # DyLin: Custom analysis
