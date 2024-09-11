from typing import Any, Callable, Dict, Tuple
from .base_analysis import BaseDyLinAnalysis
from dynapyt.instrument.filters import only


class StringStripAnalysis(BaseDyLinAnalysis):
    def __init__(self, **kwargs):
        super(StringStripAnalysis, self).__init__(**kwargs)
        self.analysis_name = "StringStripAnalysis"

    @only(patterns=["strip"])
    def post_call(
        self,
        dyn_ast: str,
        iid: int,
        val: Any,
        function: Callable,
        pos_args: Tuple,
        kw_args: Dict,
    ) -> Any:
        # print(f"{self.analysis_name} post_call {iid}")
        if val is function:
            return
        _self = getattr(function, "__self__", lambda: None)

        if not isinstance(_self, str):
            return

        if len(pos_args) > 0 and not _self is None and function == _self.strip:
            arg = pos_args[0]
            _self = function.__self__
            if len(set(arg)) != len(arg):
                self.add_finding(
                    iid,
                    dyn_ast,
                    "A-20",
                    f"Possible misuse of str.strip, arg contains duplicates {arg}",
                )
            if len(arg) > 1 and (
                (_self.startswith(arg) and _self[len(arg) : len(arg) + 1] in arg)
                or (_self.endswith(arg) and _self[-len(arg) - 1 : -len(arg)] in arg)
            ):
                self.add_finding(
                    iid,
                    dyn_ast,
                    "A-19",
                    f"Possible misuse of str.strip, might have removed something not expected before: {_self} arg: {arg} after: {val}",
                )
