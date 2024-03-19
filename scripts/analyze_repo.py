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

    if hasattr(args, "config") and args.config is not None:
        with open(args.config, "r") as f:
            config_content = f.read()
        analyses = config_content.strip().split("\n")
    else:
        analyses = [
            f"dylin.analyses.{a}.{a}"
            for a in [
                "ComparisonBehaviorAnalysis",
                "InPlaceSortAnalysis",
                "InvalidComparisonAnalysis",
                "MutableDefaultArgsAnalysis",
                "StringConcatAnalysis",
                "WrongTypeAddedAnalysis",
                "BuiltinAllAnalysis",
                "ChangeListWhileIterating",
                "StringStripAnalysis",
                "NonFinitesAnalysis",
                # Analyses below require tensorflow, pytorch, scikit-learn dependencies
                "GradientAnalysis",
                "TensorflowNonFinitesAnalysis",
                "InconsistentPreprocessing",
            ]
        ] + [
            f"dylin.analyses.ObjectMarkingAnalysis.ObjectMarkingAnalysis;config={a}"
            for a in [
                "/Work/DyLin/src/dylin/markings/configs/forced_order.yml",
                "/Work/DyLin/src/dylin/markings/configs/leak_preprocessing.yml",
                "/Work/DyLin/src/dylin/markings/configs/leaked_data.yml",
                # "/Work/DyLin/src/dylin/markings/configs/weak_hash.yml",
            ]
        ]

    if tests.endswith(".py"):
        entry = f"{name}/dylin_run_all_tests.py"
    else:
        entry = f"{name}/{tests}/dylin_run_all_tests.py"

    code_args = {'name': name, 'tests': tests}
    run_all_tests = '''
import pytest

pytest.main(['-n', 'auto', '--dist', 'worksteal', '--timeout=300', '--import-mode=importlib', '{name}/{tests}'])'''.format(
        # pytest.main(['--cov={name}', '--import-mode=importlib', '{name}/{tests}'])'''.format(
        **code_args
    )
    if name == "rich":
        analyses.remove("dylin.analyses.GradientAnalysis.GradientAnalysis")
        analyses.remove("dylin.analyses.TensorflowNonFinitesAnalysis.TensorflowNonFinitesAnalysis")

    #with open(entry, "w") as f:
    #    f.write(run_all_tests)
    #if tests.endswith(".py"):
    #    sys.path.append(str(Path(name).resolve()))
    #else:
    #    sys.path.append(str((Path(name).resolve()) / tests))
    #print("Wrote test runner, starting analysis")
    start = time.time()
    run_analysis(entry, analyses, coverage=True, coverage_dir="/Work/reports", output_dir="/Work/reports", script=run_all_tests)
    analysis_time = time.time() - start
    # print("Finished analysis, copying coverage")
    # shutil.copy("/tmp/dynapyt_coverage/covered.jsonl", "/Work/reports/")
    with open("/Work/reports/timing.txt", "a") as f:
        f.write(f"{analysis_time}\n")
