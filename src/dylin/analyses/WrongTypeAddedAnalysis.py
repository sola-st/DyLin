import types
from .base_analysis import BaseDyLinAnalysis
from dynapyt.instrument.filters import only
from typing import Any, Callable, Tuple, Dict
import random

"""
Name: 
Wrong type added

Source:
-

Test description:
Iff a list is sufficiently large and only contains objects of the same type,
adding one of another type can mean an underlying issue

Why useful in a dynamic analysis approach:
Impossible for static anylsis

Discussion:
How large is sufficiently large? N=1000?
"""

function_names = ["append", "extend", "insert", "add"]


class WrongTypeAddedAnalysis(BaseDyLinAnalysis):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nmb_add = 0
        self.nmb_add_assign = 0
        self.nmb_functions = 0
        self.analysis_name = "WrongTypeAddedAnalysis"
        self.threshold = 10

    @only(patterns=function_names)
    def pre_call(self, dyn_ast: str, iid: int, function: Callable, pos_args, kw_args):
        # print(f"{self.analysis_name} pre_call {iid}")
        if isinstance(function, types.BuiltinFunctionType) and function.__name__ in function_names:
            list_or_set = function.__self__

            if not "__len__" in dir(list_or_set) or len(list_or_set) <= self.threshold:
                return
            self.nmb_functions += 1

            type_to_check = type(random.choice(list(list_or_set)))

            # optimization to reduce overhead for large lists sample size has to be lower than threshold
            list_or_set = random.sample(list(list_or_set), self.threshold)
            same_type = all(isinstance(n, type_to_check) for n in list_or_set)

            if same_type:
                type_ok = True
                if function.__name__ in ["append", "add"]:
                    type_ok = isinstance(pos_args[0], type_to_check)
                    if not type_ok:
                        odd_type = type(pos_args[0])
                elif function.__name__ == "extend":
                    sample = pos_args[0]
                    if "__len__" in dir(sample) and len(sample) >= self.threshold:
                        sample = random.sample(list(pos_args[0]), self.threshold)
                    type_ok = all(isinstance(n, type_to_check) for n in sample)
                    if not type_ok:
                        odd_type = [type(n) for n in sample]
                elif function.__name__ == "insert":
                    type_ok = isinstance(pos_args[1], type_to_check)
                    if not type_ok:
                        odd_type = type(pos_args[1])

                if not type_ok:
                    self.add_finding(
                        iid,
                        dyn_ast,
                        "A-11",
                        f"added potentially wrong type {odd_type} to list of type {type_to_check} in {dyn_ast}",
                    )

    def add_assign(self, dyn_ast: str, iid: int, left: Any, right: Any) -> Any:
        # for some reason left is a lambda
        # print(f"{self.analysis_name} += {iid}")
        self.add(dyn_ast, iid, left(), right)

    def add(self, dyn_ast: str, iid: int, left: Any, right: Any, result: Any = None) -> Any:
        # print(f"{self.analysis_name} + {iid}")
        if isinstance(left, list):
            if len(left) <= self.threshold:
                return

            self.nmb_add += 1

            type_to_check = type(right)
            if type_to_check == list and len(right) > 0:
                type_to_check = type(right[0])
            else:
                return
            one_type = type(random.choice(left))
            random_subset = random.sample(left, self.threshold)
            homogeneous = all(isinstance(n, one_type) for n in random_subset)
            same_type = all(isinstance(n, type_to_check) for n in random_subset)

            # before addition types where the same, if not after addition we may have a problem
            if homogeneous and not same_type:
                self.add_finding(
                    iid,
                    dyn_ast,
                    "A-11",
                    f"add potentially wrong type {type_to_check} to a list of types {one_type}",
                )

    def end_execution(self) -> None:
        self.add_meta(
            {
                "add": self.nmb_add,
                "add_assign": self.nmb_add_assign,
                "nmb_interesing_functions": self.nmb_functions,
            }
        )
        super().end_execution()
