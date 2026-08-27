import logging
import os
from typing import Any, Dict, Optional
import sys
import json
import csv
import uuid
from pathlib import Path
from dynapyt.analyses.BaseAnalysis import BaseAnalysis
from dynapyt.instrument.IIDs import Location
import traceback
from filelock import FileLock


class BaseDyLinAnalysis(BaseAnalysis):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.unique_id = str(uuid.uuid4())
        self.unique_findings = set()
        self.findings = {}
        self.number_findings = 0
        self.meta = {}
        self.stack_levels = 20
        self.path = Path(self.output_dir)
        self.path.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(stream=sys.stderr)
        self.log = logging.getLogger("TestsuiteWrapper")
        self.log.setLevel(logging.DEBUG)

    def setup(self):
        # Hook for subclasses
        pass

    def add_finding(
        self,
        iid: int,
        filename: str,
        name: Optional[str] = "placeholder name",
        msg: Optional[str] = None,
    ) -> None:
        finding_key = (iid, filename, name)
        if finding_key in self.unique_findings:
            return
        self.unique_findings.add(finding_key)
        self.number_findings += 1
        stacktrace = "".join(traceback.format_stack(limit=self.stack_levels))
        location = self.iid_to_location(filename, iid)
        if name not in self.findings:
            self.findings[name] = [self._create_error_msg(iid, location, stacktrace, msg)]
        else:
            self.findings[name].append(self._create_error_msg(iid, location, stacktrace, msg))

    def get_result(self) -> Any:
        findings = self._format_issues(self.findings)
        if len(findings) == 0:
            return None
        return {
            self.analysis_name: {
                "nmb_findings": self.number_findings,
                "is_sane": self.is_sane(),
                "meta": self.meta,
                "results": findings,
            }
        }

    """
    sanity check to make sure all findings are added properly
    """

    def is_sane(self) -> bool:
        return self.number_findings == sum(len(v) for v in self.findings.values())

    def add_meta(self, meta: any):
        self.meta = meta

    def _create_error_msg(
        self,
        iid: int,
        location: Location,
        stacktrace: Optional[str] = None,
        msg: Optional[str] = None,
    ) -> Any:
        return {
            "msg": msg,
            "trace": stacktrace,
            "location": location._asdict(),
            "uid": str(location),
            "iid": iid,
            "test_id": os.environ.get("PYTEST_CURRENT_TEST"),
        }

    def _format_issues(self, findings: Dict) -> Dict:
        res = {}
        for name in findings:
            found_iids = {}
            for finding in findings[name]:
                if not finding["uid"] in found_iids:
                    found_iids[finding["uid"]] = {"finding": finding, "n": 1}
                else:
                    found_iids[finding["uid"]]["n"] += 1
            res[name] = list(found_iids.values())
        return res

    def _write_detailed_results(self):
        temp_res = self.get_result()
        if temp_res is not None:
            result = {"meta": self.meta, "results": temp_res}
            filename = f"output-{str(self.analysis_name)}-{self.unique_id}-report.json"
            with open(self.path / filename, "w") as report:
                json.dump(result, report, indent=2)

    def _write_overview(self):
        # prevent reporting findings multiple times to the same iid
        results = self._format_issues(self.findings)
        row_findings = sum(len(results[f_name]) for f_name in results)
        csv_file = self.path / "findings.csv"
        with FileLock(str(csv_file) + ".lock"):
            rows_dict = {}
            if csv_file.exists():
                with open(csv_file, "r") as f:
                    for row in csv.reader(f):
                        if row:
                            rows_dict[row[0]] = int(row[1])
            rows_dict[self.analysis_name] = rows_dict.get(self.analysis_name, 0) + row_findings
            with open(csv_file, "w") as f:
                writer = csv.writer(f)
                writer.writerows([[k, v] for k, v in rows_dict.items()])

    def end_execution(self) -> None:
        self._write_detailed_results()
        self._write_overview()
