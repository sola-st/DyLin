# ============================== Define spec ==============================
from .base_analysis import BaseDyLinAnalysis
from dynapyt.instrument.filters import only

from typing import Callable, Tuple, Dict


"""
    This specification ensures that the elements of an array are comparable before sorting them.
    Source: https://docs.python.org/3/library/functions.html#sorted.
"""


class Arrays_Comparable(BaseDyLinAnalysis):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.analysis_name = "Arrays_Comparable"

    @only(patterns=["sorted"])
    def pre_call(
        self, dyn_ast: str, iid: int, function: Callable, pos_args: Tuple, kw_args: Dict
    ) -> None:
        # The target class names for monitoring
        targets = ["builtins"]

        # Get the class name
        if hasattr(function, '__module__'):
            class_name = function.__module__
        else:
            class_name = None

        # Check if the class name is the target ones
        if class_name in targets:

            # Spec content
            objs = pos_args[0]
            if isinstance(objs, list):
                new_objs = objs[:]  # Shallow copy the elements in the list inputted.
                if kw_args.get('key'):  # If a key method for comparison is provided.
                    key = kw_args['key']  # Store the key method.
                    for i in range(len(new_objs)):  # Convert the elements using the inputted key method.
                        new_objs[i] = key(new_objs[i])
                try:  # Check if the object is comparable.
                    for i in range(len(new_objs)):
                        for j in range(i + 1, len(new_objs)):
                            # This will raise a TypeError if elements at i and j are not comparable.
                            _ = new_objs[i] < new_objs[j]
                except TypeError:
                    self.add_finding(
                        iid,
                        dyn_ast,
                        "B-1",
                        f"Array with non-comparable elements is about to be sorted at {dyn_ast}."
                    )
# =========================================================================
