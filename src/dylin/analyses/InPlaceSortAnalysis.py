import traceback
from .base_analysis import BaseDyLinAnalysis
from typing import Any, Callable, Dict, Tuple
from dynapyt.instrument.filters import only

"""
Name: 
UseInplaceSorting

Source:
-

Test description:
Inplace sorting is much faster if a copy is not needed

Why useful in a dynamic analysis approach:
No corresponding static analysis found and we can check if for some runs the
reference to the unsorted list is not needed, indicating for some cases it might be 
useful to skip the sorted() method and do in place sorting.

Discussion:


"""


class InPlaceSortAnalysis(BaseDyLinAnalysis):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.analysis_name = "InPlaceSortAnalysis"
        self.stored_lists = {}
        self.threshold = 1000

    @only(patterns=["sorted"])
    def pre_call(self, dyn_ast: str, iid: int, function: Callable, pos_args, kw_args) -> Any:
        # print(f"{self.analysis_name} pre_call {iid}")
        if function is sorted:
            # we have to keep the list in memory to keep id(pos_args[0]) stable ? nope!
            if hasattr(pos_args[0], "__len__") and len(pos_args[0]) > self.threshold:
                self.stored_lists[id(pos_args[0])] = {
                    "iid": iid,
                    "file_name": dyn_ast,
                    "len": len(pos_args[0]),
                }

    def read_identifier(self, dyn_ast: str, iid: int, val: Any) -> Any:
        # print(f"{self.analysis_name} read id {iid} {dyn_ast}")
        if len(self.stored_lists) > 0 and type(val) is list:
            self.stored_lists.pop(id(val), None)
        return None

    def end_execution(self) -> None:
        for _, l in self.stored_lists.items():
            self.add_finding(
                l["iid"],
                l["file_name"],
                "A-09",
                f"unnessecary use of sorted(), len:{l['len']} in {l['file_name']}",
            )
        super().end_execution()
