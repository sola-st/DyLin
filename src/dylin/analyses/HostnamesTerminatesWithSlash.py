# ============================== Define spec ==============================
from .base_analysis import BaseDyLinAnalysis
from dynapyt.instrument.filters import only

from typing import Callable, Tuple, Dict


"""
    It is recommended to terminate full hostnames with a /.
"""


class HostnamesTerminatesWithSlash(BaseDyLinAnalysis):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.analysis_name = "HostnamesTerminatesWithSlash"

    @only(patterns=["mount"])
    def pre_call(
        self, dyn_ast: str, iid: int, function: Callable, pos_args: Tuple, kw_args: Dict
    ) -> None:
        # The target class names for monitoring
        targets = ["requests.sessions.Session"]

        # Get the class name
        if hasattr(function, '__self__') and hasattr(function.__self__, '__class__'):
            cls = function.__self__.__class__
            class_name = cls.__module__ + "." + cls.__name__
        else:
            class_name = None

        # Check if the class name is the target ones
        if class_name in targets:

            # Spec content
            url = pos_args[0]  # Updated to use the first argument as self is not considered here
            if not url.endswith('/'):

                # Spec content
                self.add_finding(
                    iid,
                    dyn_ast,
                    "B-6",
                    f"The call to method mount in file at {dyn_ast} does not terminate the hostname with a /."
                )
# =========================================================================
