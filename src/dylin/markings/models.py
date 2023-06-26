from typing import Callable, Dict, List, Optional, Set, Tuple


class Marking():
    def __init__(self, name: str):
        self.name = name

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return other.name == self.name
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.name)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name


class StoredElement():
    def __init__(self, markings: List[Marking], location: Tuple[int, str]):
        # consider using a dict, is faster
        self.markings = set(markings)
        self.location = location

    def add_marking(self, marking: Marking):
        self.markings.append(marking)

    def remove_marking(self, marking: Marking):
        self.markings = filter(lambda m: m.marking != marking.name)

    def contains_marking(self, marking: Marking):
        return marking in self.markings

    def __repr__(self) -> str:
        return f"<Stored Element markings:{self.markings} location: {self.location}>"


def union(input: List[Set[Marking]], associated: Set[Marking] = None) -> Set[Marking]:
    if not associated:
        associated = set()

    res = associated
    for i in input:
        res = res | i
    return res

def clear(input: List[Set[Marking]], associated: Set[Marking] = None):
    if not associated:
        return set()
    res = set()
    for m in input:
        for m_l in m:
            if not m_l in associated:
                res.add(m_l)
    return res


def disjunctive_union(input: List[Set[Marking]], associated: Set[Marking] = None) -> Set[Marking]:
    if not associated:
        associated = set()
    res = set()
    for x in input:
        for y in x:
            if not y in associated:
                res.add(y)
    return res


def contains(input: Dict[str, Set[Marking]],
                associated: Set[Marking],
                argnames: List[str] = list()) -> bool:
    for m_l in associated:
        for m in input.values():
            if m_l in m:
                return True
    return False


def contains_all(input: Dict[str, Set[Marking]],
                associated: Set[Marking],
                argnames: List[str] = list()) -> bool:
    containsAll = True
    for m_l in associated:
        for m in input.values():
            if not m_l in m:
                containsAll = False
    if len(input.values()) == 0:
        return False
    return containsAll

def first_contains_all(input: Dict[str, Set[Marking]],
                associated: Set[Marking],
                argnames: List[str] = list()) -> bool:
    new_in = dict(list(input.items())[:1])
    return contains_all(new_in, associated, argnames)


def not_all_given_args_contain(input: Dict[str, Set[Marking]],
                               associated: Set[Marking],
                               argnames: List[str]) -> bool:
    all_argvals_contain = True
    for i_val in input:
        if i_val in argnames and input[i_val] != associated:
            all_argvals_contain = False
    return not all_argvals_contain

def none_contain(input: Dict[str, Set[Marking]],
                associated: Set[Marking],
                argnames: List[str] = list()) -> bool:
    for a_m in associated:
        for i in input.values():
            if a_m in i:
                return False
    return True

def not_all_or_none_contains(input: Dict[str, Set[Marking]],
          associated: Set[Marking],
          argnames: List[str] = list()) -> bool:
    all = contains_all(input, associated, argnames)
    none_contains = none_contain(input, associated, argnames)
    return not (all or none_contains)

class Source():
    '''
    TODO allow setting markings to specific output values,
    parameters
    '''

    def __init__(self,
                 associated: Set[Marking],
                 function: Callable = union,
                 assign_to_output=True,
                 assign_to_self=False):
        self.associated_markings = associated
        self.function = function
        # makes sure to assign output markings to returned objects
        self.assign_to_output = assign_to_output
        # assignes output markings to object itself
        self.assign_to_self = assign_to_self

    def get_output_markings(self, input_markings: List[Set[Marking]]) -> Set[Marking]:
        return self.function.__call__(input_markings, self.associated_markings)


class Sink():
    def __init__(self,
                 associated: Set[Marking],
                 error_msg: str,
                 argnames: List[str] = None,
                 validate: Callable = contains_all):
        self.associated_markings = associated
        self.argnames = argnames
        self.validate = validate
        self.error_msg = error_msg

    def _get_argname(self, index: int) -> str:
        if index > len(self.argnames)-1:
            return str(index)
        return self.argnames[index]

    def get_result(self, input_markings: List[Set[Marking]]) -> Optional[str]:
        input_args = {}
        argnames = []
        i = 0
        # allows optional setting of argnames, argnames not specified
        # but present in method signature simply get their arg index as
        # name
        for arg_marking in input_markings:
            input_args[self._get_argname(i)] = arg_marking
            argnames.append(self._get_argname(i))
            i += 1
        if self.validate.__call__(input_args, self.associated_markings, argnames):
            return self.error_msg
        return None


class TaintConfig():
    def __init__(self,
                 sources: Dict[str, Source],
                 sinks: Dict[str, Sink],
                 markings: Dict[str, Marking]):
        self.sources = sources
        self.sinks = sinks
        self.markings = markings
