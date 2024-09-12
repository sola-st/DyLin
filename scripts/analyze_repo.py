import argparse
import sys
from pathlib import Path
import subprocess
import shutil
from dynapyt.run_instrumentation import instrument_dir
from dynapyt.run_analysis import run_analysis
from dynapyt.post_run import post_run
import os
import signal
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a git repo")
    parser.add_argument("--repo", help="the repo index", type=int)
    parser.add_argument("--config", help="DyLin config file path", type=str)
    parser.add_argument("--cov", help="Whether to collect coverage", default=True, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    session_id = "1234-abcd"

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
pytest.main(['-s', '--timeout=300', '--import-mode=importlib', '{name}/{tests}'])'''.format(
# pytest.main(['-o', 'log_cli=true', '-n', 'auto', '--dist', 'worksteal', '--timeout=300', '--import-mode=importlib', '{name}/{tests}'])'''.format(
        # pytest.main(['--cov={name}', '--import-mode=importlib', '{name}/{tests}'])'''.format(
        **code_args
    )
    command_to_run = ["pytest", '-n', 'auto', '--dist', 'worksteal', '--import-mode=importlib', f'{name}/{tests}']
    # command_to_run = ["pytest", '-s', '--import-mode=importlib', f'{name}/{tests}']
    if name in ["rich", "python_future", "requests", "keras"]:
        analyses.remove("dylin.analyses.GradientAnalysis.GradientAnalysis")
        analyses.remove("dylin.analyses.TensorflowNonFinitesAnalysis.TensorflowNonFinitesAnalysis")
    if name == "keras":
        os.environ["KERAS_HOME"] = "/Work/DyLin/scripts"
        command_to_run = "pytest -n auto --dist worksteal keras --ignore keras/src/applications".split(" ")
    if url == "https://github.com/dpkp/kafka-python.git":
        command_to_run = ['pytest', '-n', 'auto', '--dist', 'worksteal', '--timeout=300', '--import-mode=importlib', '/Work/kafka_python/test']
    if name == "steam-market":
        command_to_run = f"pytest -n auto --dist worksteal {name}/tests.py".split(" ")
#         subprocess.run(["ls", "/opt/dylinVenv/lib/python3.10/site-packages/"])
#         run_all_tests = '''
# import pytest

# pytest.main(['-o', 'log_cli=true', '-n', 'auto', '--dist', 'worksteal', '--timeout=300', '--import-mode=importlib', '/Work/kafka_python/test'])'''
#         run_all_tests = '''
# import subprocess
# subprocess.run(["tox", "-c", "./kafka_python/tox.ini"])
#         '''

    #with open(entry, "w") as f:
    #    f.write(run_all_tests)
    #if tests.endswith(".py"):
    #    sys.path.append(str(Path(name).resolve()))
    #else:
    #    sys.path.append(str((Path(name).resolve()) / tests))
    #print("Wrote test runner, starting analysis")
    with open(f"/tmp/dynapyt_analyses-{session_id}.txt", "w") as f:
        f.write("\n".join([f"{ana};output_dir=/tmp/dynapyt_output-{session_id}" for ana in analyses]))
    os.environ["DYNAPYT_SESSION_ID"] = session_id
    timeout_threshold = 60*60
    timed_out = False
    if args.cov:
        os.environ["DYNAPYT_COVERAGE"] = f"/tmp/dynapyt_coverage-{session_id}"
        start = time.time()
        try:
            # subprocess.run(command_to_run)
            proc = subprocess.Popen(command_to_run, start_new_session=True)
            proc.wait(timeout_threshold)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        # session_id = run_analysis(entry, analyses, coverage=True, coverage_dir="/Work/reports", output_dir="/Work/reports", script=run_all_tests)
        analysis_time = time.time() - start
        post_run(coverage_dir=f"/tmp/dynapyt_coverage-{session_id}", output_dir=f"/tmp/dynapyt_output-{session_id}")
    else:
        if "DYNAPYT_COVERAGE" in os.environ:
            del os.environ["DYNAPYT_COVERAGE"]
        start = time.time()
        try:
            # subprocess.run(command_to_run)
            proc = subprocess.Popen(command_to_run, start_new_session=True)
            proc.wait(timeout_threshold)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        # session_id = run_analysis(entry, analyses, coverage=False, output_dir="/Work/reports", script=run_all_tests)
        analysis_time = time.time() - start
        post_run(output_dir=f"/tmp/dynapyt_output-{session_id}")
    # print("Finished analysis, copying coverage")
    # shutil.copy("/tmp/dynapyt_coverage/covered.jsonl", "/Work/reports/")
    with open("/Work/reports/timing.txt", "a") as f:
        f.write(f"{analysis_time} {'timed out' if timed_out else ''}\n")
