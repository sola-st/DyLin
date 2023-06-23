from typing import Any, Iterable, Iterator, List, Optional
from .base_analysis import BaseDyLinAnalysis
import collections


class ChangeListWhileIterating(BaseDyLinAnalysis):
    class ListMeta:
        def __init__(self, l: Iterable, length: int, iid: int, warned: bool = False):
            self.l = l
            self.length = length
            self.warned = warned
            self.iid = iid

    def __init__(self):
        super(ChangeListWhileIterating, self).__init__()
        self.analysis_name = "ChangeListWhileIterating"
        self.iterator_stack: List[self.ListMeta] = []

    def enter_for(
        self, dyn_ast: str, iid: int, next_value: Any, iterable: Iterable
    ) -> Optional[Any]:
        if isinstance(iterable, collections.abc.Iterator) or isinstance(
            iterable, type({})
        ):
            return

        _list = iterable
        try:
            if len(self.iterator_stack) == 0 or (
                len(_list) != len(self.iterator_stack[-1].l)
                and _list != self.iterator_stack[-1].l
            ):
                length = len(_list)

                self.iterator_stack.append(self.ListMeta(_list, length, iid))
            elif len(self.iterator_stack) > 0:
                list_meta: self.ListMeta = self.iterator_stack[-1]
                if (
                    list_meta.warned is False
                    and iid == list_meta.iid
                    and len(_list) != list_meta.length
                    and id(_list) == id(list_meta.l)
                    and iterable == self.iterator_stack[-1].l
                ):
                    self.add_finding(
                        iid,
                        dyn_ast,
                        "A-22",
                        f"List length changed while iterating initial length: {list_meta.length} current:{len(_list)}",
                    )
                    list_meta.warned = True
        # necessary for dynamically loaded lists during runtime which sometimes can not be compared in certain
        # text cases
        except Exception as e:
            print(e)

    def exit_for(self, dyn_ast, iid):
        if len(self.iterator_stack) > 0:
            self.iterator_stack.pop()
