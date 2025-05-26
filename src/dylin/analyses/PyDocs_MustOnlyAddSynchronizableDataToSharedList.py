# ============================== Define spec ==============================
from .base_analysis import BaseDyLinAnalysis
from dynapyt.instrument.filters import only

from typing import Callable, Tuple, Dict
from multiprocessing.shared_memory import SharedMemory
import socket


"""
    Must only add synchronizable data to shared list.
"""


def is_synchronizable(data):
    # If it's a dict, it's not synchronizable
    if isinstance(data, dict):
        return False
    
    # If it's a list, it's not synchronizable
    if isinstance(data, list):
        return False

    # SharedMemory objects are not synchronizable
    if isinstance(data, SharedMemory):
        return False
    
    # socket objects are not synchronizable
    if isinstance(data, socket.socket):
        return False


class PyDocs_MustOnlyAddSynchronizableDataToSharedList(BaseDyLinAnalysis):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.analysis_name = "PyDocs_MustOnlyAddSynchronizableDataToSharedList"

    @only(patterns=["append"])
    def pre_call(
        self, dyn_ast: str, iid: int, function: Callable, pos_args: Tuple, kw_args: Dict
    ) -> None:
        # The target class names for monitoring
        targets = ["multiprocessing.managers.ListProxy"]

        # Get the class name
        if hasattr(function, '__self__') and hasattr(function.__self__, '__class__'):
            cls = function.__self__.__class__
            class_name = cls.__module__ + "." + cls.__name__
        else:
            class_name = None

        # Check if the class name is the target ones
        if class_name in targets:

            # Get the data
            data = None
            if kw_args.get('object'):
                data = kw_args['object']
            elif pos_args:
                data = pos_args[0]

            # Check if the data is synchronizable
            if not is_synchronizable(data):

                # Spec content
                self.add_finding(
                    iid,
                    dyn_ast,
                    "B-11",
                    f"Must only add synchronizable data to shared list at {dyn_ast}."
                )
# =========================================================================
