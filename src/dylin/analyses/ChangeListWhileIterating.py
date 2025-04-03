from typing import Any, Iterable, Iterator, List, Optional
from .base_analysis import BaseDyLinAnalysis
import collections


class ChangeListWhileIterating(BaseDyLinAnalysis):
    # PyLint has a check for this, code W4701
    # PyLint also has checks for dictionaries and sets (E4702, E4703). These seem more severe than the list check
    class ListMeta:
        def __init__(self, l: Iterable, length: int, dyn_ast: str, iid: int, warned: bool = False):
            self.l = l
            self.length = length
            self.warned = warned
            self.dyn_ast = dyn_ast
            self.iid = iid

    def __init__(self, **kwargs):
        super(ChangeListWhileIterating, self).__init__(**kwargs)
        self.analysis_name = "ChangeListWhileIterating"
        self.iterator_stack: List[self.ListMeta] = []

    def enter_for(self, dyn_ast: str, iid: int, next_value: Any, iterable: Iterable) -> Optional[Any]:
        # print(f"{self.analysis_name} enter_for {iid}")
        if isinstance(iterable, collections.abc.Iterator) or isinstance(iterable, type({})):
            return

        _list = iterable
        try:
            if (
                len(self.iterator_stack) == 0
                or iid != self.iterator_stack[-1].iid
                or dyn_ast != self.iterator_stack[-1].dyn_ast
            ):
                length = len(_list)
                self.iterator_stack.append(self.ListMeta(_list, length, dyn_ast, iid))
            elif len(self.iterator_stack) > 0:
                list_meta: self.ListMeta = self.iterator_stack[-1]
                if (
                    list_meta.warned is False
                    and len(_list) < list_meta.length
                    and id(_list) == id(list_meta.l)
                    and iterable == list_meta.l
                ):
                    self.add_finding(
                        iid,
                        dyn_ast,
                        "A-22",
                        f"List length changed while iterating initial length: {list_meta.length} current:{len(_list)}",
                    )
                    list_meta.warned = True
        # necessary for dynamically loaded lists during runtime which sometimes can not be compared in certain
        # test cases
        except Exception as e:
            print(e)

    def exit_for(self, dyn_ast, iid):
        # print(f"{self.analysis_name} exit_for {iid}")
        if len(self.iterator_stack) > 0:
            self.iterator_stack.pop()
