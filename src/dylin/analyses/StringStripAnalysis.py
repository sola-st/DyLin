from typing import Any, Callable, Dict, Tuple
from .base_analysis import BaseDyLinAnalysis


class StringStripAnalysis(BaseDyLinAnalysis):
    def __init__(self):
        super(StringStripAnalysis, self).__init__()
        self.analysis_name = "StringStripAnalysis"

    def post_call(
        self,
        dyn_ast: str,
        iid: int,
        val: Any,
        function: Callable,
        pos_args: Tuple,
        kw_args: Dict,
    ) -> Any:
        _self = getattr(function, "__self__", lambda: None)

        if not isinstance(_self, str):
            return

        if len(pos_args) > 0 and not _self is None and function == _self.strip:
            arg = pos_args[0]
            _self = function.__self__
            if len(set(arg)) != len(arg):
                if not arg in _self:
                    self.add_finding(
                        iid,
                        dyn_ast,
                        "A-19",
                        f"Possible misuse of str.strip, might have removed something not expected before: {_self} arg: {arg} after: {val}",
                    )
                else:
                    self.add_finding(
                        iid,
                        dyn_ast,
                        "A-20",
                        f"Possible misuse of str.strip, arg contains duplicates {arg}",
                    )
