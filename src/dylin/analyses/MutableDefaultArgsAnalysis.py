from .base_analysis import BaseDyLinAnalysis
from typing import Any, Callable, List, Optional, Tuple, Dict
import traceback

"""
Name: 
Mutable Default Arguments

Source:
-

Test description:
Because methods are first class objects default values are saved. If default values are changed
a second invocation of the function will keep the changed value

Why useful in a dynamic analysis approach:
Non trivial control flows can make it nearly impossible to detect such behavior in static analysis

Discussion:
This can be useful and is intended behavior for python. Experienced python programmers might use this
deliberately.
"""


class MutableDefaultArgsAnalysis(BaseDyLinAnalysis):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.function_calls = {}
        self.analysis_name = "MutableDefaultArgsAnalysis"

    def pre_call(self, dyn_ast: str, iid: int, function: Callable, pos_args, kw_args):
        dicts_and_lists = self.get_dicts_and_lists_as_str(function)
        # we are only interested in [] or {} or set()
        if dicts_and_lists is None:
            return

        if function not in self.function_calls:
            # comparing string representations of default values is sufficient here
            # and more performant compared to a deep list comparison of each element
            self.function_calls[function] = {
                "defaults": dicts_and_lists,
                "name": function.__name__,
            }
        else:
            if self.function_calls[function]["defaults"] != dicts_and_lists:
                prev = self.function_calls[function]["defaults"]
                self.add_finding(
                    iid,
                    dyn_ast,
                    "A-10",
                    f"mutable default args reused and changed in function {function.__name__} args: after: {dicts_and_lists} \n prev: {prev} in {dyn_ast}",
                )

    def get_dicts_and_lists_as_str(self, function: Callable) -> Optional[str]:
        # built in methods do not have defaults
        if not "__defaults__" in dir(function):
            return

        defaults = function.__defaults__
        if defaults is None:
            return None

        res = []
        for i in defaults:
            if isinstance(i, type([])) or isinstance(i, type({})) or isinstance(i, type(set())):
                res.append(i)
        if len(res) == 0:
            return None
        return str(res)
