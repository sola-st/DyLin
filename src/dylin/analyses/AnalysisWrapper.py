from typing import Any, Callable, Iterator, List, Dict, Optional, Tuple, Union
from ctypes import Array
import importlib
import os
from pathlib import Path
from typing import Any, Callable
import json
from .base_analysis import BaseDyLinAnalysis
import csv


class AnalysisWrapper(BaseDyLinAnalysis):
    def __init__(self) -> None:
        self.analysis_name = "AnalysisWrapper"
        self.analysis_classes: Array[BaseDyLinAnalysis] = []
        self.log_msgs: List[str] = []
        classNames = [
            "FilesClosedAnalysis",
            "ComparisonBehaviorAnalysis",
            "InPlaceSortAnalysis",
            "SideEffectsDunderAnalysis",
            "InvalidComparisonAnalysis",
            "MutableDefaultArgsAnalysis",
            "StringConcatAnalysis",
            "WrongTypeAddedAnalysis",
            "BuiltinAllAnalysis",
            "ChangeListWhileIterating",
            "StringStripAnalysis",
            "NonFinitesAnalysis",
            # Analyses below require tensorflow, pytorch, scikit-learn dependencies
            "GradientAnalysis",
            "TensorflowNonFinitesAnalysis",
            "InconsistentPreprocessing",
        ]
        self.metadata = None
        here = Path(__file__).parent.resolve()
        self.path = Path("/Work", "reports")
        self.analysis_name = None

        # TODO workaround, make this dynamic later
        self.number_unique_findings_possible = 33

        for name in classNames:
            module = importlib.import_module("dylin.analyses." + name)
            cls = getattr(module, name)()
            if cls is not None:
                self.analysis_classes.append(cls)
            else:
                raise ValueError(f"class with name {name} not found")

        configs_path = here / ".." / "markings" / "configs"

        files = [f for f in configs_path.iterdir() if f.is_file()]
        for file in files:
            module = importlib.import_module("dylin.analyses.ObjectMarkingAnalysis")
            cls: BaseDyLinAnalysis = getattr(module, "ObjectMarkingAnalysis")()
            cls.add_meta({"configName": str(file)})
            cls.setup()
            if cls is not None:
                self.analysis_classes.append(cls)
            else:
                raise ValueError(f"class with name {name} not found")

    def _write_detailed_results(self):
        collect_dicts = []
        for cls in self.analysis_classes:
            collect_dicts.append(cls.get_result())
        result = {"meta": self.metadata, "results": collect_dicts}
        filename = str(self.analysis_name) + "report.json"
        # collect_dicts.append({"log": self.log_msgs})
        with open(self.path / filename, "a+") as report:
            report.write(json.dumps(result, indent=4))

    def _write_overview(self):
        row_findings = [0] * self.number_unique_findings_possible
        for cls in self.analysis_classes:
            # prevent reporting findings multiple times to the same iid
            results = cls.get_unique_findings()
            for f_name in results:
                col_index = f_name.split("-")[-1]
                row_findings[int(col_index) - 1] = len(results[f_name])

        csv_row = [self.analysis_name] + row_findings
        with open(self.path / "findings.csv", "a+") as f:
            writer = csv.writer(f)
            writer.writerow(csv_row)

    def call_if_exists(self, f: str, *args) -> Any:
        for c in self.analysis_classes:
            func: Callable = getattr(c, f, lambda *args: None)(*args)
            if func is not None:
                self.log_msgs.append(f"ignored return value from {func.__name__} in class {c}")
        return None

    def add_metadata(self, meta: any) -> None:
        self.metadata = meta
        self.analysis_name = self.metadata.get("name")
        if self.analysis_name is None:
            self.analysis_name = "No name set"

    def end_execution(self) -> None:
        self.call_if_exists("end_execution")
        self._write_detailed_results()
        self._write_overview()

    def runtime_event(self, dyn_ast: str, iid: int) -> None:
        if self.analysis_name is None:
            self.analysis_name = dyn_ast.split("/")[-1]

    def read_attribute(self, dyn_ast, iid, base, name, val):
        return self.call_if_exists("read_attribute", dyn_ast, iid, base, name, val)

    def pre_call(self, dyn_ast: str, iid: int, function: Callable, pos_args, kw_args):
        return self.call_if_exists("pre_call", dyn_ast, iid, function, pos_args, kw_args)

    def post_call(
        self,
        dyn_ast: str,
        iid: int,
        val: Any,
        function: Callable,
        pos_args: Tuple,
        kw_args: Dict,
    ) -> Any:
        return self.call_if_exists("post_call", dyn_ast, iid, val, function, pos_args, kw_args)

    def comparison(self, dyn_ast: str, iid: int, left: Any, op: str, right: Any, result: Any) -> bool:
        return self.call_if_exists("comparison", dyn_ast, iid, left, op, right, result)

    def equal(self, dyn_ast: str, iid: int, left: Any, right: Any, result: Any) -> bool:
        return self.call_if_exists("equal", dyn_ast, iid, left, right, result)

    def not_equal(self, dyn_ast: str, iid: int, left: Any, right: Any, result: Any) -> bool:
        return self.call_if_exists("not_equal", dyn_ast, iid, left, right, result)

    def add_assign(self, dyn_ast: str, iid: int, left: Any, right: Any) -> Any:
        return self.call_if_exists("add_assign", dyn_ast, iid, left, right)

    def add(self, dyn_ast: str, iid: int, left: Any, right: Any, result: Any = None) -> Any:
        return self.call_if_exists("add", dyn_ast, iid, left, right, result)

    def write(self, dyn_ast: str, iid: int, old_val: Any, new_val: Any) -> Any:
        return self.call_if_exists("write", dyn_ast, iid, old_val, new_val)

    def read_identifier(self, dyn_ast: str, iid: int, val: Any) -> Any:
        return self.call_if_exists("read_identifier", dyn_ast, iid, val)

    def function_enter(self, dyn_ast: str, iid: int, args: List[Any], name: str, is_lambda: bool) -> None:
        return self.call_if_exists("function_enter", dyn_ast, iid, args, name, is_lambda)

    def function_exit(self, dyn_ast: str, iid: int, function_name: str, result: Any) -> Any:
        return self.call_if_exists("function_exit", dyn_ast, iid, function_name, result)

    def _list(self, dyn_ast: str, iid: int, value: List) -> List:
        return self.call_if_exists("_list", dyn_ast, iid, value)

    def binary_operation(self, dyn_ast: str, iid: int, op: str, left: Any, right: Any, result: Any) -> Any:
        return self.call_if_exists("binary_operation", dyn_ast, iid, op, left, right, result)

    def read_subscript(self, dyn_ast: str, iid: int, base: Any, sl: List[Union[int, Tuple]], val: Any) -> Any:
        return self.call_if_exists("read_subscript", dyn_ast, iid, base, sl, val)

    def enter_for(self, dyn_ast: str, iid: int, next_value: Any, iterator: Iterator) -> Optional[Any]:
        return self.call_if_exists("enter_for", dyn_ast, iid, next_value, iterator)

    def exit_for(self, dyn_ast, iid):
        return self.call_if_exists("exit_for", dyn_ast, iid)
