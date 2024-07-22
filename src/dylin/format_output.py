import json
import fire
from .select_checkers import issue_codes


def format_output(findings_path: str) -> str:
    with open(findings_path, "r") as f:
        findings = json.load(f)
    res = ""
    for finding in findings:
        if len(finding["results"]) > 0:
            for result in finding["results"]:
                if len(result["results"]) > 0:
                    for issue_code, issues in result["results"].items():
                        code = ""
                        for c, i in issue_codes.items():
                            if issue_code in i["aliases"] or issue_code == c:
                                code = c
                                break
                        for issue in issues:
                            res += f"{code}: {issue['location']['file']}: {issue['location']['start_line']}: {issue['msg']}\n"
    return res


if __name__ == "__main__":
    fire.Fire(format_output)
