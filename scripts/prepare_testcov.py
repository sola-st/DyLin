import argparse
import sys
from pathlib import Path
import subprocess
import shutil
from dynapyt.run_instrumentation import instrument_dir
from dynapyt.run_analysis import run_analysis
import os
import time
from common import install_special

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def post_process_special(url):
    if url == "https://github.com/pallets/click.git":
        (Path("click").resolve() / "tests" / "test_imports.py").unlink(missing_ok=True)


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
    install_special(url)
    print("Installed special requirements")

    if url.startswith("http"):
        subprocess.run(["sh", here / "get_repo.sh", url, commit, name])
        print("Cloned repo and switched to commit")
        if requirements:
            subprocess.run(["pip", "install", "-r", f"{name}/{requirements}"])
        if url == "https://github.com/tiangolo/typer.git":
            subprocess.run(["pip", "install", f"{name}/[all]"])
        else:
            subprocess.run(["pip", "install", "-e", f"{name}/"])
        print("Installed requirements")
    else:
        if requirements:
            print((here/url/requirements).exists())
            subprocess.run(["pip", "install", "-r", f"{str((here/url/requirements).resolve())}"])
        print((here/url).exists())
        subprocess.run(["ls", f"{str(here)}/.."])
        subprocess.run(["pip", "install", "-e", f"{str((here/url).resolve())}/"])

    post_process_special(url)
    print("Post processed special requirements")

    if not url.startswith("http"):
        name = str((here / url).resolve())
