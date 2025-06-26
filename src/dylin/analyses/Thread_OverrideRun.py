# ============================== Define spec ==============================
from .base_analysis import BaseDyLinAnalysis
from dynapyt.instrument.filters import only

from typing import Callable, Tuple, Dict
from inspect import getsource
from hashlib import sha256
import threading


"""
    This is used to check if the run method of a Thread is overridden 
    or the argument 'target' is passed in via the constructor.
"""


class Thread_OverrideRun(BaseDyLinAnalysis):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.analysis_name = "Thread_OverrideRun"

        # Get the hash of the original run method for inspection
        try:  
            source_code = getsource(threading.Thread.run)  
            self.original_run_method_hash = sha256(source_code.encode()).hexdigest()  
        except Exception as e:  
            # Fallback: Log a warning and set a default value  
            print(f"Warning: Unable to retrieve source code for threading.Thread.run. Exception: {e}")  
            self.original_run_method_hash = None  

    @only(patterns=["start"])
    def pre_call(
        self, dyn_ast: str, iid: int, function: Callable, pos_args: Tuple, kw_args: Dict
    ) -> None:
        # The target class names for monitoring
        targets = ["threading.Thread"]

        # Get the class name
        if hasattr(function, '__self__') and hasattr(function.__self__, '__class__'):
            cls = function.__self__.__class__
            class_name = cls.__module__ + "." + cls.__name__
        else:
            class_name = None

        # Check if the class name is the target ones
        if class_name in targets:

            # Get the function object
            obj = function.__self__

            # Check if the run method is overridden
            sha = sha256(getsource(obj.run).encode()).hexdigest()
            if sha == self.original_run_method_hash:  # method run not overridden

                # argument 'target' not passed in constructor
                if not hasattr(obj, '_target') or getattr(obj, '_target') is None:

                    # Spec content
                    self.add_finding(
                        iid,
                        dyn_ast,
                        "B-20",
                        f"Thread run method not overridden or argument target not passed in constructor at {dyn_ast}."
                    )
# =========================================================================
