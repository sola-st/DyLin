import logging
from typing import Any, Dict, Optional
import sys
from dynapyt.analyses.BaseAnalysis import BaseAnalysis
from dynapyt.instrument.IIDs import Location
import traceback


class BaseDyLinAnalysis(BaseAnalysis):
    def __init__(self) -> None:
        super(BaseDyLinAnalysis, self).__init__()
        self.findings = {}
        self.number_findings = 0
        self.meta = {}
        self.analysis_name = "BaseAnalysis"
        self.stack_levels = 100

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
        self.number_findings += 1
        stacktrace = "".join(traceback.format_stack()[-self.stack_levels :])
        location = self.iid_to_location(filename, iid)
        if not name in self.findings:
            self.findings[name] = [
                self._create_error_msg(iid, location, stacktrace, msg)
            ]
        else:
            self.findings[name].append(
                self._create_error_msg(iid, location, stacktrace, msg)
            )

    def get_result(self) -> Any:
        findings = self._format_issues(self.findings)
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
        res = 0
        for name, value in self.findings.items():
            res += len(value)
        return self.number_findings == res

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
        }

    def _format_issues(self, findings: Dict) -> Dict:
        res = {}
        for name in findings:
            found_iids = {}
            for finding in findings[name]:
                if not finding["uid"] in found_iids:
                    found_iids[finding["iid"]] = {"finding": finding, "n": 1}
                else:
                    found_iids[finding["iid"]]["n"] += 1
            res[name] = list(found_iids.values())
        return res

    def get_unique_findings(self):
        return self._format_issues(self.findings)
