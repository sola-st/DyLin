import json
from fire import Fire
from pathlib import Path

def summarize_findings(results: str):
    results = Path(results)
    findings = []
    for result in results.glob("**/findings.csv"):
        with open(result, "r") as f:
            content = f.read().split("\n")
        for l in content:
            if not l.strip().endswith(",0"):
                analysis_name = l.strip().split(",")[0]
                if len(analysis_name) == 0:
                    continue
                with open(result.parent.parent/"timing.txt", "r") as f:
                    project_name = f.read().strip().split(" ")[0]
                with open(result.parent/f"{analysis_name}report.json", "r") as f:
                    report = json.load(f)
                for finding in report["results"]:
                    for k, v in finding.items():
                        if v["nmb_findings"] > 0:
                            for code, ress in v["results"].items():
                                for r in ress:
                                    fnd = r["finding"]
                                    findings.append(f"{fnd['location']['file'][6:-5]}:{fnd['location']['start_line']}:{fnd['location']['start_column']}: {code} {fnd['msg']}")
    with open(results/"DyLin_findings.txt", "w") as f:
        f.write("\n".join(findings))

if __name__ == "__main__":
    Fire(summarize_findings)