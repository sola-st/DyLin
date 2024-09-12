import argparse
import json
from pathlib import Path
import subprocess


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a git repo")
    parser.add_argument("--repo", help="the repo index", type=int)
    parser.add_argument("--config", help="DyLin config file path", type=str)
    args = parser.parse_args()

    here = Path(__file__).parent.resolve()
    with open(here / "projects.txt", "r") as f:
        project_infos = f.read().split("\n")

    project_infos = [p for p in project_infos if not p.startswith("#")]

    project_info = project_infos[args.repo - 1].split(" ")
    if "r" in project_info[2]:
        url, commit, flags, requirements, tests = tuple(project_info)
    else:
        url, commit, flags, tests = tuple(project_info)
        requirements = None
    name = url.split("/")[-1].split(".")[0].replace("-", "_")
    print("Extracted repo info")

    if url.startswith("http"):
        subprocess.run(["sh", here / "get_repo.sh", url, commit, name])
        print("Cloned repo and switched to commit")

    if not url.startswith("http"):
        name = str((here / url).resolve())

    result = subprocess.run(
        ["ruff", "check", "--select", "ALL", "--output-format", "concise", name], stdout=subprocess.PIPE
    )

    with open("/Work/lint_reports/results_ruff.txt", "w") as f:
        f.write(result.stdout.decode("utf-8"))
    
    result = subprocess.run(
        ["pylint", "-j", "0", name], stdout=subprocess.PIPE
    )

    with open("/Work/lint_reports/results_pylint.txt", "w") as f:
        f.write(result.stdout.decode("utf-8"))
    
    result = subprocess.run(
        ["mypy", name], stdout=subprocess.PIPE
    )

    with open("/Work/lint_reports/results_mypy.txt", "w") as f:
        f.write(result.stdout.decode("utf-8"))
