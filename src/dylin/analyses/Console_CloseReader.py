# ============================== Define spec ==============================
from .base_analysis import BaseDyLinAnalysis
from dynapyt.instrument.filters import only

from typing import Callable, Tuple, Dict


"""
    This specification warns if close() is invoked on sys.stdin which is a useless invocation.
    Source: https://docs.python.org/3/faq/library.html#why-doesn-t-closing-sys-stdout-stdin-stderr-really-close-it.
"""


class Console_CloseReader(BaseDyLinAnalysis):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.analysis_name = "Console_CloseReader"

    @only(patterns=["close"])
    def pre_call(
        self, dyn_ast: str, iid: int, function: Callable, pos_args: Tuple, kw_args: Dict
    ) -> None:
        # The target class names for monitoring
        targets = ["<stdin>"]

        # Get the class name
        if hasattr(function, '__self__') and hasattr(function.__self__, 'name'):
            class_name = function.__self__.name
        else:
            class_name = None

        # Check if the class name is the target ones
        if class_name in targets:

            # Spec content
            self.add_finding(
                iid,
                dyn_ast,
                "B-3",
                f"close() is invoked on sys.stdin which is a useless invocation at {dyn_ast}."
            )
# =========================================================================
