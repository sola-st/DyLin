from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

from libcst import Tuple
from .base_analysis import BaseDyLinAnalysis
from ..markings.obj_identifier import uniqueid, save_uid, get_ref, add_cleanup_hook
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator


class InconsistentPreprocessing(BaseDyLinAnalysis):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.analysis_name = "InconsistentPreprocessing"
        # Note: a boolean here is actually enough, we use a set though to allow
        # more detailed analyses in the future
        self.markings_storage = defaultdict(set)

        def cleanup(x: any):
            if x in self.markings_storage:
                self.markings_storage[x] = set()

        add_cleanup_hook(lambda x: cleanup(x))

    def read_subscript(self, dyn_ast, iid, base, sl, val):
        if save_uid(base) in self.markings_storage and len(self.markings_storage[save_uid(base)]) != 0:
            self.markings_storage[uniqueid(val)].add("transformed")

    def read_attribute(self, dyn_ast, iid, base, name, val):
        if save_uid(base) in self.markings_storage and len(self.markings_storage[save_uid(base)]) != 0:
            self.markings_storage[uniqueid(val)].add("transformed")

    def post_call(
        self,
        dyn_ast: str,
        iid: int,
        result: Any,
        function: Callable,
        pos_args: Tuple,
        kw_args: Dict,
    ) -> Any:
        if result is function:
            return
        try:
            func_name = function.__name__
        except AttributeError:
            return
        _self = getattr(function, "__self__", lambda: None)
        if _self is None:
            return

        in_args = list(kw_args.values() if not kw_args is None else []) + list(pos_args if not pos_args is None else [])

        in_args.append(_self)

        if function.__name__ == "fit_transform" and isinstance(_self, TransformerMixin):
            # source
            self.markings_storage[uniqueid(result)].add("transformed")
            self.markings_storage[uniqueid(_self)].add("transformed")
        elif function.__name__ == "transform" and isinstance(_self, TransformerMixin):
            # source
            self.markings_storage[uniqueid(result)].add("transformed")
        elif function.__name__ == "fit" and isinstance(_self, TransformerMixin):
            # source
            self.markings_storage[uniqueid(_self)].add("transformed")
        elif function.__name__ == "predict" and isinstance(_self, BaseEstimator):
            # sink
            in_args = list(pos_args if not pos_args is None else []) + [_self]

            count = len(in_args)
            for arg in in_args:
                if save_uid(arg) in self.markings_storage and len(self.markings_storage[save_uid(arg)]) > 0:
                    count -= 1

            if count != 0 and count != len(in_args):
                self.add_finding(
                    iid,
                    dyn_ast,
                    "M-23",
                    f"{count} args have not been transformed out of {len(in_args)} --> {[self.markings_storage.get(save_uid(arg), 'not') for arg in in_args]}",
                )

        else:
            # propagate marking
            is_arg_marked = any(
                [
                    (save_uid(arg) in self.markings_storage and len(self.markings_storage[save_uid(arg)]) > 0)
                    or (
                        (isinstance(arg, list) or isinstance(arg, tuple))
                        and any(
                            save_uid(a) in self.markings_storage and len(self.markings_storage[save_uid(a)]) > 0
                            for a in arg
                        )
                    )
                    for arg in in_args
                ]
            )

            if is_arg_marked and function.__name__ == "fit" and isinstance(_self, BaseEstimator):
                self.markings_storage[uniqueid(_self)].add("transformed")

            if not type(result) is tuple and not type(result) is list:
                if not result is None and is_arg_marked:
                    self.markings_storage[uniqueid(result)].add("transformed")
                    # self.markings_storage[uniqueid(_self)].add("transformed")
            else:
                for r in result:
                    if save_uid(r) in self.markings_storage:
                        is_result_stored = len(self.markings_storage[save_uid(r)]) > 0
                        if not r is None and is_arg_marked and not is_result_stored:
                            self.markings_storage[uniqueid(result)].add("transformed")
                    else:
                        if r is not None and is_arg_marked:
                            self.markings_storage[uniqueid(r)] = set(["transformed"])
