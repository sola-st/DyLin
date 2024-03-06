from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .base_analysis import BaseDyLinAnalysis
from ..markings import models
from ..markings.obj_identifier import uniqueid, cleanup, save_uid, has_obj
import yaml
import json


class ObjectMarkingAnalysis(BaseDyLinAnalysis):
    """
    We only care about values on the heap here.
    Furthermore, we define that values can only be tainted if some function has been
    called on them.

    But this is not entirely true, consider the following:
        @HTTP.POST("/usr")
        def createUser(username: string)
            ...
    Username should obviously be tainted. TODO we can easily work around this in the future
    with function_entry / exit, which allows to taint input values during runtime, based
    on the method name.

    Ideas for identifying objects: ids are not stable across execution, also they (probably) only work
    like this in cpython, other implementations might not refer to memory addresses for id()
    calls
    ideas:
        - get current stack frame and associate StoredElements with id and frame
            - if frame is gone, gc StoredElements
            -> inspect.get_frame is really slow and still cpython specific
        - keep weakreference via weakref module of value
            - if its been garbage collected / freed, weakref just returns None
            - implement "gc" to clean up weakrefs which return None
            - again, does not work for some python interpreters, but might be faster
        - wrap every object during instrumentation with a special wrapper which is able to relay every function call
            - (see https://stackoverflow.com/questions/33125419/instance-of-python-class-that-responds-to-all-method-calls)
            - comes with its own set of problems, i.e. external objects from libraries, builtins etc etc.
            - probaly creates all kinds of issues during execution

        current solution:
            - use id() but remove objects dynamically via callback of weakrefs as soon as
              object is removed by gc
                - if developers define __slots__ themselves w/o __weakref__ we can't index object
            - wrap dicts, lists etc in a wrapper class during runtime which contains
              __weakref__ in slots, also for developer defined classes which overwrite __slots__ w/o
              weakref (this is can be done via _write_)
                - evaluate how often this breaks code bases i.e. type(strWrap) != str
            - does not work for int, tuple. Hash objects in that case

    Furthermore, the current implementation does not consider something like this:
        if value < 2:
            other_value.do_sth_dangerous()
    where value impacts dataflow of other_value

    TODO: add binary etc operations to be treated as normal operation
    """

    def __init__(self, **kwargs):
        self.analysis_name = "ObjectMarkingAnalysis"
        self.stored_elements = {}
        self.sources = {}
        self.sinks = {}
        self.log = []
        if "config" in kwargs:
            self.load_config(kwargs["config"])
            del kwargs["config"]
        else:
            raise ValueError(f"no config set for ObjectMarkingAnalysis {kwargs}")
        super().__init__(**kwargs)

    def setup(self):
        config_name: str = self.meta.get("configName")
        if not config_name.endswith(".yml"):
            config_name += ".yml"

        if not config_name:
            raise ValueError("no config set for ObjectMarkingAnalysis")

        import pathlib

        pwd = pathlib.Path(__file__).parent.resolve()
        configPath = pwd / ".." / "markings" / "configs" / config_name

        self.load_config(str(configPath))

    def load_config(self, yaml_path: str) -> models.TaintConfig:
        with open(yaml_path, "r") as yaml_str:
            yml = yaml.safe_load(yaml_str)

        name = yml.get("name")
        if name:
            self.analysis_name = name

        # load markings
        markings = {}
        for key, marking in yml["markings"].items():
            markings[key] = models.Marking(marking["name"])

        # load sources
        sources = {}
        for key in yml["sources"]:
            m = set()
            source = yml["sources"][key]
            for m_key in source["associated_markings"]:
                m.add(markings[m_key])
            f_string = source.get("function")
            resulting_function = None
            if f_string == "disjunctive_union":
                resulting_function = models.disjunctive_union
            elif f_string == "clear":
                resulting_function = models.clear
            else:
                resulting_function = models.union

            if "qualnames" in source:
                for qualname in source["qualnames"]:
                    sources[qualname] = models.Source(m, resulting_function)
            else:
                sources[key] = models.Source(m, resulting_function)

        # load sinks
        sinks = {}
        for key in yml["sinks"]:
            m = set()
            sink = yml["sinks"][key]

            # refrenced markings
            for m_key in sink["associated_markings"]:
                m.add(markings[m_key])

            # validation function
            f_string = sink["validate"]["name"]
            resulting_function = None
            if f_string == "contains_all":
                resulting_function = models.contains_all
            elif f_string == "contains":
                resulting_function = models.contains
            elif f_string == "not_all_given_args_contain":
                resulting_function = models.not_all_given_args_contain
            elif f_string == "not_all_or_none_contains":
                resulting_function = models.not_all_or_none_contains
            elif f_string == "first_contains_all":
                resulting_function = models.first_contains_all
            # error message
            error_msg = sink["error_msg"]

            # args
            args = list()
            for arg in sink.get("args") if sink.get("args") else list():
                args.append(arg)

            if "qualnames" in sink:
                for qualname in sink["qualnames"]:
                    sinks[qualname] = models.Sink(m, error_msg, args, resulting_function)
            else:
                sinks[key] = models.Sink(m, error_msg, args, resulting_function)

        self.sources = sources
        self.sinks = sinks

    # returns markings as a list of sets, where the outer list preserves
    # the order of arguments of the original method signature
    # if method contains __self__ its the first argument, if not we ignore it
    def _get_in_markings(self, pos_args: Tuple, kw_args: Dict, _self: Optional[Any]) -> List[Set[models.Marking]]:
        in_markings = []
        if not _self is None:
            element = self.stored_elements.get(save_uid(_self))
            if element:
                in_markings.append(element.markings)
        if not pos_args is None:
            for arg in pos_args:
                element = self.stored_elements.get(save_uid(arg))
                if element:
                    in_markings.append(element.markings)
                else:
                    in_markings.append(set())
        if not kw_args is None:
            for arg in kw_args:
                element = self.stored_elements.get(save_uid(arg))
                if element:
                    in_markings.append(element.markings)
                else:
                    in_markings.append(set())
        return in_markings

    def post_call(
        self,
        dyn_ast: str,
        iid: int,
        result: Any,
        function: Callable,
        pos_args: Tuple,
        kw_args: Dict,
    ) -> Any:
        function_is_of_interest = False
        # TODO: use __module__ + '.' + __qualname__ to receive fully qualified name
        func_name = getattr(function, "__qualname__", lambda: None)

        if func_name is None:
            return None
        if func_name in self.sources:
            source: models.Source = self.sources[func_name]
            _self = getattr(function, "__self__", lambda: None)
            out_markings = source.get_output_markings(self._get_in_markings(pos_args, kw_args, _self))

            # TODO store markings to arguments as well if desired
            if source.assign_to_output and not result is None:
                if type(result) is tuple:
                    for r in result:
                        self.stored_elements[uniqueid(r)] = models.StoredElement(out_markings, (iid, None, func_name))
                else:
                    self.stored_elements[uniqueid(result)] = models.StoredElement(out_markings, (iid, None, func_name))

            if source.assign_to_self and not _self is None:
                self.stored_elements[uniqueid(_self)] = models.StoredElement(out_markings, (iid, None, func_name))
            function_is_of_interest = True
        if func_name in self.sinks:
            sink: models.Sink = self.sinks[func_name]
            _self = getattr(function, "__self__", lambda: None)
            error = sink.get_result(self._get_in_markings(pos_args, kw_args, _self))
            if error:
                _selfstr = str(self.stored_elements.get(save_uid(_self)))
                self.add_finding(iid, dyn_ast, error, _selfstr)
            function_is_of_interest = True
        if not function_is_of_interest and not result is None and len(self.stored_elements) > 0:
            _self = getattr(function, "__self__", lambda: None)
            # default for functions: union of input markings
            out_markings_result = models.union(self._get_in_markings(pos_args, kw_args, _self))

            """
            Consider the following case:
            def a(x):
                x.removes_marking()
                return x
            a(marks_x(x))

            Normally the returned object of a would also contain the marking
            because by default we use the union operation.
            However, in this case the function a which is instrumented as
            well takes care of the markings. This is not the case
            for non instrumented parts of the source code, e.g. libraries,
            c-source files etc. is_result_stored solves this case
            """

            if not type(result) is tuple and not type(result) is list:
                is_result_stored = self.stored_elements.get(save_uid(result)) != None

                if not result is None and len(out_markings_result) > 0 and not is_result_stored:
                    self.stored_elements[uniqueid(result)] = models.StoredElement(
                        out_markings_result, (iid, None, func_name)
                    )
            else:
                i = 0
                for r in result:
                    is_result_stored = self.stored_elements.get(save_uid(result)) != None
                    if not r is None and len(out_markings_result) > 0 and not is_result_stored:
                        self.stored_elements[uniqueid(r)] = models.StoredElement(
                            out_markings_result, (iid, None, str(func_name) + str(i))
                        )
                        i = i + 1
        return None

    def function_exit(self, dyn_ast: str, iid: int, name: str, result: Any) -> Any:
        # TODO ignore return value of function which calls cleanup
        cleanup()

    def end_execution(self) -> None:
        self.add_meta(f"stored elements {len(self.stored_elements)}, {list(self.stored_elements.values())[:100]}")
        super().end_execution()
