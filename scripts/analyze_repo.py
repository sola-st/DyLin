import argparse
import sys
from pathlib import Path
import subprocess

from dynapyt.run_instrumentation import instrument_dir
from dynapyt.run_analysis import run_analysis


def install_special(url):
    if url == "https://github.com/lorien/grab.git":
        command = "pip install cssselect pyquery pymongo fastrq"  # required for running tests
    elif url == "https://github.com/psf/black.git":
        command = "pip install aiohttp"  # required for running tests
    elif url == "https://github.com/errbotio/errbot.git":
        command = "pip install mock"  # required for running tests
    elif url == "https://github.com/PyFilesystem/pyfilesystem2.git":
        command = "pip install parameterized pyftpdlib psutil"  # required for running tests
    elif url == "https://github.com/wtforms/wtforms.git":
        command = "pip install babel email_validator"  # required for running tests
    elif url == "https://github.com/geopy/geopy.git":
        command = "pip install docutils"  # required for running tests
    elif url == "https://github.com/gawel/pyquery.git":
        command = "pip install webtest"  # required for running tests
    elif url == "https://github.com/elastic/elasticsearch-dsl-py.git":
        command = "pip install pytz"  # required for running tests
    elif url == "https://github.com/marshmallow-code/marshmallow.git":
        command = "pip install pytz simplejson"  # required for running tests
    elif url == "https://github.com/pytest-dev/pytest.git":
        command = "pip install hypothesis xmlschema"  # required for running tests
    else:
        return
    subprocess.run(command.split(" "))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a git repo")
    parser.add_argument("--repo", help="the repo index", type=int)
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
    install_special(url)

    subprocess.run(["sh", here / "get_repo.sh", url, commit, name])
    if requirements:
        subprocess.run(["pip", "install", "-r", f"{name}/{requirements}"])
    subprocess.run(["pip", "install", f"{name}/"])

    installation_dir = f"/opt/dylinVenv/lib/python3.10/site-packages/{name}"
    analyses = [
        f"dylin.analyses.{a}.{a}"
        for a in [
            "FilesClosedAnalysis",
            "ComparisonBehaviorAnalysis",
            "InPlaceSortAnalysis",
            "SideEffectsDunderAnalysis",
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
    ]
    instrument_dir(installation_dir, analyses, use_external_dir=False)
    instrument_dir(name, analyses, use_external_dir=False)
    if tests.endswith(".py"):
        entry = f"{name}/dylin_run_all_tests.py"
    else:
        entry = f"{name}/{tests}/dylin_run_all_tests.py"

    code_args = {'name': name, 'tests': tests, 'analyses': repr(analyses)}
    run_all_tests = '''
import pytest

class AnalysisSetupPlugin:
    def pytest_xdist_node_collection_finished(self, node, ids):
        import dynapyt.runtime as rt
        rt.set_analysis({analyses})
    def pytest_collection_modifyitems(self, items):
        import dynapyt.runtime as rt
        rt.set_analysis({analyses})

pytest.main(['-n', 'auto', '--dist', 'worksteal', '--import-mode=importlib', '{name}/{tests}'], plugins=[AnalysisSetupPlugin()])'''.format(
        **code_args
    )
    with open(entry, "w") as f:
        f.write(run_all_tests)
    if tests.endswith(".py"):
        sys.path.append(str(Path(name).resolve()))
    else:
        sys.path.append(str((Path(name).resolve()) / tests))
    run_analysis("dylin_run_all_tests", analyses)

    Path("/Work", "reports", "report.json").rename(f"/Work/reports/report_{name}.json")
    Path("/Work", "reports", "findings.csv").rename(f"/Work/reports/findings_{name}.csv")
