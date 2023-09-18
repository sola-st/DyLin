from os import path
from .base_analysis import BaseDyLinAnalysis
from typing import Any, Callable, List
import inspect


class SideEffectsDunderAnalysis(BaseDyLinAnalysis):
    def __init__(self, **kwargs):
        super(SideEffectsDunderAnalysis, self).__init__(**kwargs)
        self.analysis_name = "SideEffectsDunderAnalysis"
        self.stack_levels = 20
        self.dunder_method_stack: List[Callable] = []
        self.cached_file_contents = {}
        self.dunder_methods_to_check = {
            # equality and hashing
            "__eq__",
            "__ge__",
            "__gt__",
            "__le__",
            "__ne__",
            "__lt__",
            "__hash__",
            # binary operators
            "__and__",
            "__divmod__",
            "__floordiv__",
            "__lshift__",
            "__matmul__",
            "__mod__",
            "__mul__",
            "__or__",
            "__pow__",
            "__sub__",
            "__xor__",
            "__radd__",
            "__rand__",
            "__rdiv__",
            "__rdivmod__",
            "__rfloordiv_",
            "__rlshift__",
            "__rmatmul__",
            "__rmod__",
            "__rmul__",
            "__ror__",
            "__rpow__",
            "__rrshift__",
            "__rsub__",
            "__rtruediv__",
            "__rxor__",
            "__iadd__",
            "__iand__",
            "__ifloordiv_",
            "__ilshift__",
            "__imatmul__",
            "__imod__",
            "__imul__",
            "__ior__",
            "__ipow__",
            "__irshift__",
            "__isub__",
            "__itruediv__",
            "__ixor__",
            # unary operators
            "__abs__",
            "__neg__",
            "__pos__",
            "__invert__",
            # math
            "__index__",
            "__trunc__",
            "__floor__",
            "__ceil__",
            "__round__",
            # iterator
            "__len__",
            "__reversed__",
            # numeric type casting
            "__int__",
            "__bool__",
            "__nonzero__",
            "__complex__",
            "__float__",
            # others
            "__format__",
            "__cmp__",
            "__str__",
            "__repr__",
            # instance / subclass checks
            "__instancecheck__",
            "__subclasscheck__",
        }

    def function_enter(self, dyn_ast: str, iid: int, args: List[Any], name: str, is_lambda: bool) -> None:
        if name in self.dunder_methods_to_check:
            self.dunder_method_stack.append(name)

    # TODO does not work, old_vals can contain _read_ which uses external variables
    # Consider calling dunder method twice, compare state of object after first call and after second call,
    # nothing should not have changed
    def write(self, dyn_ast: str, iid: int, old_vals: List[Callable], new_val: Any) -> None:
        if len(self.dunder_method_stack) > 0:
            try:
                closure_vars = inspect.getclosurevars(old_vals[0])
            except ValueError:
                return
            if closure_vars.globals and self._check_stack_sanity(self.dunder_method_stack):
                self.add_finding(
                    iid,
                    dyn_ast,
                    "A-06",
                    f"wrote to global variable in {self.dunder_method_stack[-1]} global vars: {closure_vars.globals}",
                )
            elif "self" in closure_vars.nonlocals.keys() and self._check_stack_sanity(self.dunder_method_stack):
                pass
                # self.add_finding(iid, dyn_ast, "A-07", f"wrote to attribute in {self.dunder_method_stack[-1]} attribute {closure_vars.nonlocals.keys()}")

    """
    Checks if internal stack is still sane. If a crash / exception occured dunder_method_stack might
    be not in sync -> clear internal stack if thats the case
    """

    def _check_stack_sanity(self, dunder_method_stack):
        if len(dunder_method_stack) > 0:
            function_name = dunder_method_stack[-1]
            method_name_stack = list(map(lambda frame_info: frame_info.function, inspect.stack()))
            res = function_name in method_name_stack
            if not res:
                self.dunder_method_stack = []
            return res
        return False

    def _check_if_left_method(self, dyn_ast: str, iid: int, function_name: str):
        if not str(function_name) in self.dunder_methods_to_check:
            return

        if len(self.dunder_method_stack) > 0 and self.dunder_method_stack[-1] == function_name:
            self.dunder_method_stack.pop()

    def function_exit(self, dyn_ast: str, iid: int, name: str, result: Any) -> Any:
        self._check_if_left_method(dyn_ast, iid, name)
        return None
