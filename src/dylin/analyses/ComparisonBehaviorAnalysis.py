from typing import Any
import operator
from .base_analysis import BaseDyLinAnalysis
import numpy as np


class ComparisonBehaviorAnalysis(BaseDyLinAnalysis):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.analysis_name = "ComparisonBehaviorAnalysis"
        self.excluded_types = [type(0.0), type(None)]
        self.stack_levels = 20
        self.cache = {}

    """
    TODO:
    don't check for all comparisons only if
    "__eq__",
    "__ge__",
    "__gt__",
    "__le__",
    "__ne__",
    "__lt__",
    are implemented by hand
    """

    def is_excluded(self, val: any) -> bool:
        return isinstance(val, type(0.0)) or isinstance(val, type(None)) or isinstance(val, np.floating) or isinstance(val, np.ndarray)

    def equal(self, dyn_ast: str, iid: int, left: Any, right: Any, result: Any) -> bool:
        self.check_all(dyn_ast, iid, left, "Equal", right, result)

    def not_equal(self, dyn_ast: str, iid: int, left: Any, right: Any, result: Any) -> bool:
        self.check_all(dyn_ast, iid, left, "NotEqual", right, result)

    def check_all(self, dyn_ast: str, iid: int, left: Any, op: str, right: Any, result: Any) -> bool:
        op_function = None
        if op == "Equal":
            op_function = operator.eq
        elif op == "NotEqual":
            op_function = operator.ne
        else:
            return None

        if self.is_excluded(left) or self.is_excluded(right):
            return None

        try:
            if self.check_symmetry(left, right, op_function, result):
                self.add_finding(
                    iid,
                    dyn_ast,
                    "A-01",
                    f"bad symmetry for {op_function} with {left} {right}",
                )
            if self.check_stability(left, right, op_function, result):
                self.add_finding(iid, dyn_ast, "A-02", f"bad stability for {op_function}")
            elif self.check_identity(left, op_function):
                self.add_finding(
                    iid,
                    dyn_ast,
                    "A-03",
                    f"bad identity {op_function} of {left} returned true when compared with None",
                )
            elif self.check_reflexivity(left, op_function):
                self.add_finding(
                    iid,
                    dyn_ast,
                    "A-04",
                    f"bad reflexivity {left} {op_function} to itself",
                )
        except ValueError:
            # some libraries e.g. pandas do not allow to do all kinds of comparisons e.g. pandas.series == None
            return

    def check_reflexivity(self, left, op: Any) -> bool:
        if op == operator.ne:
            # 3 != 3
            return op(left, left)
        # not(3 == 3)
        return not op(left, left)

    def check_identity(self, left: Any, op: Any) -> bool:
        if op == operator.ne:
            # not(None == 3)
            return not op(left, None)
        # None == 3
        return op(left, None)

    def check_symmetry(self, left: Any, right: Any, op: Any, res: bool) -> bool:
        # (3 == 4) == (4 == 3)
        return not (op(left, right) == op(right, left) == res)

    def check_stability(self, left: Any, right: Any, op: Any, normal: bool) -> bool:
        for i in range(3):
            if op(left, right) != normal:
                return True
        return False
