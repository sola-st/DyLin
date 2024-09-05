import json
import fire
from .select_checkers import issue_codes


def format_output(findings_path: str) -> str:
    with open(findings_path, "r") as f:
        findings = json.load(f)
    res = set()
    for finding in findings:
        if len(finding["results"]) > 0:
            for _, checker_finding in finding["results"].items():
                if len(checker_finding["results"]) > 0:
                    for issue_code, issue_list in checker_finding["results"].items():
                        code = ""
                        for c, i in issue_codes.items():
                            if issue_code in i["aliases"] or issue_code == c:
                                code = c
                                break
                        for issue in issue_list:
                            res.add(
                                f"{code}: {issue['finding']['location']['file']}: {issue['finding']['location']['start_line']}: {issue['finding']['msg']}"
                            )
    return "\n".join(list(res))


if __name__ == "__main__":
    fire.Fire(format_output)
