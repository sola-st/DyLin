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
        return (
            type(val) is int
            or type(val) is float
            or type(val) is str
            or type(val) is list
            or type(val) is set
            or type(val) is dict
            or type(val) is bool
            or isinstance(val, type(0.0))
            or isinstance(val, type(None))
            or isinstance(val, np.floating)
            or isinstance(val, np.ndarray)
        )

    def equal(self, dyn_ast: str, iid: int, left: Any, right: Any, result: Any) -> bool:
        # print(f"{self.analysis_name} equal {iid}")
        self.check_all(dyn_ast, iid, left, "Equal", right, result)

    def not_equal(self, dyn_ast: str, iid: int, left: Any, right: Any, result: Any) -> bool:
        # print(f"{self.analysis_name} not equal {iid}")
        self.check_all(dyn_ast, iid, left, "NotEqual", right, result)

    def check_all(self, dyn_ast: str, iid: int, left: Any, op: str, right: Any, result: Any) -> bool:
        if op != "Equal" and op != "NotEqual":
            return None

        if self.is_excluded(left) or self.is_excluded(right):
            return None

        try:
            if self.check_symmetry(left, right, op, result):
                self.add_finding(
                    iid,
                    dyn_ast,
                    "A-01",
                    f"bad symmetry for {op} with {left} {right}",
                )
            if self.check_stability(left, right, op, result):
                self.add_finding(iid, dyn_ast, "A-02", f"bad stability for {op}")
            elif self.check_identity(left):
                self.add_finding(
                    iid,
                    dyn_ast,
                    "A-03",
                    f"bad identity {op} of {left} returned true when compared with None",
                )
            elif self.check_identity(right):
                self.add_finding(
                    iid,
                    dyn_ast,
                    "A-03",
                    f"bad identity {op} of {right} returned true when compared with None",
                )
            elif self.check_reflexivity(left):
                self.add_finding(
                    iid,
                    dyn_ast,
                    "A-04",
                    f"bad reflexivity {left} {op} to itself",
                )
            elif self.check_reflexivity(right):
                self.add_finding(
                    iid,
                    dyn_ast,
                    "A-04",
                    f"bad reflexivity {right} {op} to itself",
                )
        except ValueError:
            # some libraries e.g. pandas do not allow to do all kinds of comparisons e.g. pandas.series == None
            return

    def check_reflexivity(self, left) -> bool:
        return left != left

    def check_identity(self, left: Any) -> bool:
        return left is not None and left == None

    def check_symmetry(self, left: Any, right: Any, op: Any, res: bool) -> bool:
        # (3 == 4) == (4 == 3)
        if op == "Equal":
            return not ((left == right) == (right == left) == res)
        else:
            return not ((left != right) == (right != left) == res)

    def check_stability(self, left: Any, right: Any, op: Any, normal: bool) -> bool:
        for _ in range(3):
            if op == "Equal":
                if (left == right) != normal:
                    return True
            else:
                if (left != right) != normal:
                    return True
        return False
