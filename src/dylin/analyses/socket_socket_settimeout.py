# ============================== Define spec ==============================
from .base_analysis import BaseDyLinAnalysis
from dynapyt.instrument.filters import only

from typing import Callable, Tuple, Dict


"""
    Timeout must not be a negative number
    src: https://docs.python.org/3/library/socket.html#socket.socket.settimeout
"""


class socket_socket_settimeout(BaseDyLinAnalysis):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.analysis_name = "socket_socket_settimeout"

    @only(patterns=["settimeout"])
    def pre_call(
        self, dyn_ast: str, iid: int, function: Callable, pos_args: Tuple, kw_args: Dict
    ) -> None:
        # The target class names for monitoring
        targets = ["socket.socket"]

        # Get the class name
        if hasattr(function, '__self__') and hasattr(function.__self__, '__class__'):
            cls = function.__self__.__class__
            class_name = cls.__module__ + "." + cls.__name__
        else:
            class_name = None

        # Check if the class name is the target ones
        if class_name in targets:

            # Spec content
            timeout = None
            if "timeout" in kw_args:
                timeout = kw_args["timeout"]
            elif len(pos_args) > 0:
                timeout = pos_args[0]
            
            # Check if the timeout is a negative number
            if timeout is not None and isinstance(timeout, (int, float)) and timeout < 0:

                # Spec content
                self.add_finding(
                    iid,
                    dyn_ast,
                    "B-19",
                    f"Timeout must not be a negative number at {dyn_ast}."
                )
# =========================================================================
