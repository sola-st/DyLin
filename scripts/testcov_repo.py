import argparse
import sys
from pathlib import Path
import subprocess
import shutil
from dynapyt.run_instrumentation import instrument_dir
from dynapyt.run_analysis import run_analysis
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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

    if not url.startswith("http"):
        name = str((here / url).resolve())
    if url == "https://github.com/tiangolo/typer.git":
        installation_dir = f"{str(Path('/opt/dylinVenv/lib/python3.10/site-packages/', name))}"

    import pytest

    start_time = time.time()
    if url == "https://github.com/tiangolo/typer.git":
        pytest.main(['-n', 'auto', '--dist', 'worksteal', '--timeout=300', f'--cov={installation_dir}', '--cov-report', 'json:/Work/testcov/cov.json', '--import-mode=importlib', f'{name}/{tests}'])
    else:
        pytest.main(['-n', 'auto', '--dist', 'worksteal', '--timeout=300', f'--cov={name}', '--cov-report', 'json:/Work/testcov/cov.json', '--import-mode=importlib', f'{name}/{tests}'])
    end_time = time.time()
    with open("/Work/testcov/timing.txt", "w") as f:
        f.write(f"{name} {end_time - start_time}\n")
