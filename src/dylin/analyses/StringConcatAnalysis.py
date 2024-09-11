from ast import List
from operator import ne
from typing import Any
from .base_analysis import BaseDyLinAnalysis

"""
Name: 
Concat strings

Source:
https://peps.python.org/pep-0008/#programming-recommendations

Test description:
String concatenation via + is only (sometimes) efficient in CPython because they implemented
in place concatenation. However, other interpreters don't have this capability.
To ensure linear time string concatenation use ''.join()

Why useful in a dynamic analysis approach:
Dynamic analysis is able to infer whether plus operator is used between strings if variables are 
not constants

Discussion:

"""


class StringConcatAnalysis(BaseDyLinAnalysis):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.concats = {}
        self.adds = {}
        self.analysis_name = "StringConcatAnalysis"
        self.last_add_operation = None
        self.threshold = 10000

    def add_assign(self, dyn_ast: str, iid: int, left: Any, right: Any) -> None:
        # for some reason left is a lambda
        # print(f"{self.analysis_name} += {iid}")
        self._check(dyn_ast, iid, right)

    def _check(self, dyn_ast: str, iid: int, right: Any, result: Any = None) -> None:
        if isinstance(right, type("")):
            key = str(iid) + "_" + dyn_ast
            if key not in self.concats:
                self.concats[key] = 1
            elif self.concats[key] != -1:
                self.concats[key] += 1
            if self.concats[key] > self.threshold:
                self.add_finding(
                    iid,
                    dyn_ast,
                    "A-05",
                    "attempted to concat strings alot with + operator",
                )
                self.concats[key] = -1

    # Removed below hooks for performance reasons
    # def add(self, dyn_ast: str, iid: int, left: Any, right: Any, result: Any) -> None:
    #     if isinstance(right, type("")):
    #         self.last_add_operation = {"iid": iid, "result": result}
    #     else:
    #         self.last_add_operation = None

    # # Only a += b or a = a + b is bad not a+b+c -> add check for a = a + b
    # def write(self, dyn_ast: str, iid: int, old_val: Any, new_val: Any) -> None:
    #     if isinstance(new_val, type("")):
    #         if self.last_add_operation is not None and self.last_add_operation["iid"] == iid - 1:
    #             if new_val == self.last_add_operation["result"]:
    #                 self._check(dyn_ast, iid, new_val)
