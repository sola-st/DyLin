import importlib
from typing import Any, Callable, Iterator, List, Dict, Optional, Set, Tuple, Union
import logging
import sys
import uuid
from .base_analysis import BaseDyLinAnalysis

"""

Wraps the testsuite.
The testsuite uses python unittests and assert statements to 
make clear which parts of the code should be flagged by an analyser.

This can be done by a simple assert statement
i.e. assert someExpressionToFlag == True 
     assert someExpressionNotToFlag == False
This wrapper will make sure to return the value True to the expression if
an issue was found and false if not, thus the test fails or succeeds

A dictionary at the start of the testfile makes tells this wrapper which analyzer
to load and wrap around.

Todo:
- find possibility to integrate pytest instead of python unittests to get
  more detailed outpus
"""


class TestsuiteWrapper:
    def __init__(self) -> None:
        self.testname: Optional[str] = None
        self.analysis_class: Optional[BaseDyLinAnalysis] = None
        self.initialized: bool = False
        self.failure_messages = []
        self.dangling_failure_messages = []
        self.desired_number_failures = 0
        self.uuid_call_stack = []
        logging.basicConfig(stream=sys.stderr)
        self.log = logging.getLogger("TestsuiteWrapper")
        self.log.setLevel(logging.DEBUG)

    def call_if_exists(self, f: str, *args) -> Any:
        func = getattr(self.analysis_class, f, lambda *args: None)
        return func(*args)

    """
    handles setup, loads analyses 
    """

    def dictionary(self, dyn_ast: str, iid: int, items: List[Any], value: Dict) -> Dict:
        if not self.initialized:
            self.testname = list(value)[0]
            analysisName = value[self.testname]
            module = importlib.import_module("dylin." + analysisName)
            class_ = getattr(module, analysisName)
            current_analysis = class_()
            self.analysis_class = current_analysis
            if value.get("configName"):
                self.analysis_class.add_meta({"configName": value["configName"]})
            self.analysis_class.setup()
            self.log.debug(f"running test {self.testname}")
            self.initialized = True
        else:
            return self.call_if_exists("dictionary", value, dyn_ast, iid, items, value)
        return value

    """
    Monitors failures, they have to be surrounded by START; and END; string literals
    Suffix of END; may include failure messages
    """

    def string(self, dyn_ast: str, iid: int, val: str) -> Any:
        if val.startswith("START;"):
            self._add_potential_issue()
        elif val.startswith("END;"):
            if not self._is_current_issue_found(iid):
                self.failure_messages.append(val[len("END;") :])
        elif val.startswith("END EXECUTION;"):
            self._add_potential_issue()
            self.dangling_failure_messages.append(val[len("END EXECUTION;") :])

    def _add_potential_issue(self):
        self.uuid_call_stack.append((uuid.uuid4(), self._current_number_findings()))
        self.desired_number_failures += 1

    def _is_current_issue_found(self, iid=-1):
        if self.uuid_call_stack:
            id, nmb_findings = self.uuid_call_stack.pop()
            if nmb_findings + 1 != self._current_number_findings():
                return False
            return True
        raise RuntimeError(f"START; is probably missing before iid {iid}")

    def _current_number_findings(self) -> int:
        return self.analysis_class.number_findings if self.analysis_class else 0

    def end_execution(self) -> None:
        self.call_if_exists("end_execution")

        for issue in self.dangling_failure_messages:
            if not self._is_current_issue_found():
                self.failure_messages.append(issue)

        nmb_issues_not_found = len(self.failure_messages)
        nmb_findings = self._current_number_findings()
        for k, v in (
            self.analysis_class.findings.items() if self.analysis_class else list()
        ):
            for finding in v:
                print(f"{finding['msg']} iid: {finding['iid']}")
        self.log.debug(f"number false negatives: {nmb_issues_not_found}")

        false_pos = 0
        if (nmb_findings - self.desired_number_failures) > 0:
            false_pos = nmb_findings - self.desired_number_failures
        self.log.debug(f"number false positives: {false_pos}")

        self.log.debug(f"found {nmb_findings}/{self.desired_number_failures}")
        self.log.debug(f"failed to find messages: \n \n{self.failure_messages}")

    def read_attribute(self, dyn_ast, iid, base, name, val):
        return self.call_if_exists("read_attribute", dyn_ast, iid, base, name, val)

    def pre_call(self, dyn_ast: str, iid: int, function: Callable, pos_args, kw_args):
        return self.call_if_exists(
            "pre_call", dyn_ast, iid, function, pos_args, kw_args
        )

    def post_call(
        self,
        dyn_ast: str,
        iid: int,
        val: Any,
        function: Callable,
        pos_args: Tuple,
        kw_args: Dict,
    ) -> Any:
        return self.call_if_exists(
            "post_call", dyn_ast, iid, val, function, pos_args, kw_args
        )

    def comparison(
        self, dyn_ast: str, iid: int, left: Any, op: str, right: Any, result: Any
    ) -> bool:
        return self.call_if_exists("comparison", dyn_ast, iid, left, op, right, result)

    def add_assign(self, dyn_ast: str, iid: int, left: Any, right: Any) -> Any:
        return self.call_if_exists("add_assign", dyn_ast, iid, left, right)

    def add(
        self, dyn_ast: str, iid: int, left: Any, right: Any, result: Any = None
    ) -> Any:
        return self.call_if_exists("add", dyn_ast, iid, left, right, result)

    def write(self, dyn_ast: str, iid: int, old_val: Any, new_val: Any) -> None:
        return self.call_if_exists("write", dyn_ast, iid, old_val, new_val)

    def read_identifier(self, dyn_ast: str, iid: int, val: Any) -> Any:
        return self.call_if_exists("read_identifier", dyn_ast, iid, val)

    def function_enter(
        self, dyn_ast: str, iid: int, args: List[Any], name: str, is_lambda: bool
    ) -> None:
        return self.call_if_exists(
            "function_enter", dyn_ast, iid, args, name, is_lambda
        )

    def function_exit(self, dyn_ast: str, iid: int, name: str, result: Any) -> Any:
        return self.call_if_exists("function_exit", dyn_ast, iid, name, result)

    def _list(self, dyn_ast: str, iid: int, value: List) -> List:
        return self.call_if_exists("_list", dyn_ast, iid, value)

    def binary_operation(
        self, dyn_ast: str, iid: int, op: str, left: Any, right: Any, result: Any
    ) -> Any:
        return self.call_if_exists(
            "binary_operation", dyn_ast, iid, op, left, right, result
        )

    def read_subscript(
        self, dyn_ast: str, iid: int, base: Any, sl: List[Union[int, Tuple]], val: Any
    ) -> Any:
        return self.call_if_exists("read_subscript", dyn_ast, iid, base, sl, val)

    def enter_for(
        self, dyn_ast: str, iid: int, next_value: Any, iterator: Iterator
    ) -> Optional[Any]:
        return self.call_if_exists("enter_for", dyn_ast, iid, next_value, iterator)

    def exit_for(self, dyn_ast, iid):
        return self.call_if_exists("exit_for", dyn_ast, iid)
