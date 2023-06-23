import argparse
from pathlib import Path
import subprocess
from dynapyt.run_instrumentation import instrument_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a git repo")
    parser.add_argument("--repo", help="the repo index", type=int)
    args = parser.parse_args()

    here = Path(__file__).parent.resolve()
    with open(here / "projects.txt", "r") as f:
        project_infos = f.readlines()

    project_info = project_infos[args.repo - 1].split(" ")
    if "r" in project_info[2]:
        url, commit, flags, requirements, tests = tuple(project_info)
    else:
        url, commit, flags, tests = tuple(project_info)
        requirements = None
    name = url.split("/")[-1].split(".")[0].replace("-", "_")

    # subprocess.run(["ls", ".."])
    subprocess.run(["sh", here / "get_repo.sh", url, commit, name])
    if requirements:
        subprocess.run(["pip", "install", "-r", f"{name}/{requirements}"])
    subprocess.run(["pip", "install", f"{name}/"])

    src_dir = f"/opt/dylinVenv/lib/python3.10/site-packages/{name}"
    analysis = "AnalysisWrapper"
    instrument_dir(src_dir, analysis, module="dylin")

    # subprocess.run(["pip", "show", name])
