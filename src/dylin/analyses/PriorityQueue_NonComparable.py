# ============================== Define spec ==============================
from .base_analysis import BaseDyLinAnalysis
from dynapyt.instrument.filters import only

from typing import Callable, Tuple, Dict


"""
    This is used to check if PriorityQueue is about to have a non-comparable object.
"""


class PriorityQueue_NonComparable(BaseDyLinAnalysis):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.analysis_name = "PriorityQueue_NonComparable"

    @only(patterns=["put", "heappush"])
    def pre_call(
        self, dyn_ast: str, iid: int, function: Callable, pos_args: Tuple, kw_args: Dict
    ) -> None:
        # The target class names for monitoring
        targets = ["queue", "_heapq"]

        # Get the class name
        if hasattr(function, '__module__') and hasattr(function, '__name__'):
            class_name = function.__module__
        else:
            class_name = None

        # Check if the class name is the target ones
        if class_name in targets:

            # Spec content
            if function.__name__ == "put" and class_name == "queue" and hasattr(function, '__self__') and type(function.__self__).__name__ == "PriorityQueue":
                obj = pos_args[0]
                try:  # check if the object is comparable
                    obj < obj
                except TypeError as e:

                    # Spec content
                    self.add_finding(
                        iid,
                        dyn_ast,
                        "B-10",
                        f"PriorityQueue is about to have a non-comparable object at {dyn_ast}."
                    )

            if function.__name__ == "heappush" and class_name == "_heapq":
                obj = pos_args[1]
                try:  # check if the object is comparable
                    obj < obj
                except TypeError as e:

                    # Spec content
                    self.add_finding(
                        iid,
                        dyn_ast,
                        "B-10",
                        f"PriorityQueue is about to have a non-comparable object at {dyn_ast}."
                    )
# =========================================================================
