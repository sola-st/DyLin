from fire import Fire
from pathlib import Path

def compare(static_dir: str, dynamic: str):
    with open(dynamic, "r") as f:
        dynamic_findings = f.read().split("\n")
    dynamic_issues = {":".join(f.split(":")[:2]): ":".join(f.split(":")[2:]) for f in dynamic_findings}
    # lint_repo.py writes results_ruff.txt / results_pylint.txt / results_mypy.txt (not results.txt)
    static_files = []
    for pattern in ("**/results_ruff.txt", "**/results_pylint.txt", "**/results_mypy.txt", "**/results.txt"):
        static_files.extend(Path(static_dir).glob(pattern))
    for static in sorted(set(static_files)):
        with open(static, "r") as f:
            static_findings = f.read().split("\n")
        for sf in static_findings:
            static_location = ":".join(sf.split(":")[:2])
            if static_location in dynamic_issues:
                print(f"Static: {sf} Dynamic: {dynamic_issues[static_location]}")

if __name__ == "__main__":
    Fire(compare)