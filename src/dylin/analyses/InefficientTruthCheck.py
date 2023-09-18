from .base_analysis import BaseDyLinAnalysis
from typing import Any, Callable, List
from time import time_ns


class InefficientTruthCheck(BaseDyLinAnalysis):
    """
    Checks for slow __bool__ and __len__ functions.
    Based on Zhang, Zejun, et al. "Faster or Slower? Performance Mystery of Python Idioms Unveiled with Empirical Evidence." arXiv preprint arXiv:2301.12633 (2023).
    """

    def __init__(self, **kwargs):
        super(InefficientTruthCheck, self).__init__(**kwargs)
        self.analysis_name = "InefficientTruthCheck"
        self.start_time = []
        self.threshold = 10000000  # 10 ms

    def function_enter(self, dyn_ast: str, iid: int, args: List[Any], name: str, is_lambda: bool) -> None:
        if name in ["__bool__", "__len__"]:
            self.start_time.append((iid, name, time_ns()))

    def function_exit(self, dyn_ast: str, iid: int, name: str, result: Any) -> Any:
        time_now = time_ns()
        if name in ["__bool__", "__len__"]:
            if len(self.start_time) == 0:
                return
            top = self.start_time.pop()
            if top[0] != iid or top[1] != name:
                return
            elapsed_time = time_now - top[2]
            if elapsed_time > self.threshold:
                self.add_finding(
                    iid,
                    dyn_ast,
                    "A-23",
                    f"Slow {name} function took {elapsed_time/1000000} ms",
                )
