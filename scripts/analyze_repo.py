import argparse
import sys
from pathlib import Path
import subprocess
import shutil
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
    elif url == "https://github.com/miso-belica/sumy.git":
        subprocess.run(["pip", "install", "nltk"])
        command = "python -m nltk.downloader all"
    else:
        return
    subprocess.run(command.split(" "))


def post_process_special(url):
    if url == "https://github.com/pallets/click.git":
        (Path("click").resolve() / "tests" / "test_imports.py").unlink(missing_ok=True)


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
    print("Extracted repo info")
    install_special(url)
    print("Installed special requirements")

    subprocess.run(["sh", here / "get_repo.sh", url, commit, name])
    print("Cloned repo and switched to commit")
    if requirements:
        subprocess.run(["pip", "install", "-r", f"{name}/{requirements}"])
    subprocess.run(["pip", "install", f"{name}/"])
    print("Installed requirements")

    post_process_special(url)
    print("Post processed special requirements")

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
    ] + [
        f"dylin.analyses.ObjectMarkingAnalysis.ObjectMarkingAnalysis:{a}"
        for a in [
            "/Work/DyLin/src/dylin/analyses/markings/configs/forced_order.yml",
            "/Work/DyLin/src/dylin/analyses/markings/configs/leak_preprocessing.yml",
            "/Work/DyLin/src/dylin/analyses/markings/configs/leaked_data.yml",
            "/Work/DyLin/src/dylin/analyses/markings/configs/weak_hash.yml",
        ]
    ]

    if Path("/tmp/dynapyt_analyses.txt").exists():
        Path("/tmp/dynapyt_analyses.txt").unlink()
    with open('/tmp/dynapyt_analyses.txt', 'w') as f:
        f.write('\n'.join(analyses))
    print("Wrote analyses to file, starting instrumentation")
    instrument_dir(installation_dir, analyses, use_external_dir=False)
    instrument_dir(name, analyses, use_external_dir=False)
    print("Instrumented repo")
    if tests.endswith(".py"):
        entry = f"{name}/dylin_run_all_tests.py"
    else:
        entry = f"{name}/{tests}/dylin_run_all_tests.py"

    code_args = {'name': name, 'tests': tests, 'analyses': repr(analyses), 'installation_dir': installation_dir}
    run_all_tests = '''
import pytest

pytest.main(['-n', 'auto', '--dist', 'worksteal', '--cov={name}', '--cov={installation_dir}', '--import-mode=importlib', '{name}/{tests}'])'''.format(
        # pytest.main(['--cov={name}', '--import-mode=importlib', '{name}/{tests}'])'''.format(
        **code_args
    )

    with open(entry, "w") as f:
        f.write(run_all_tests)
    if tests.endswith(".py"):
        sys.path.append(str(Path(name).resolve()))
    else:
        sys.path.append(str((Path(name).resolve()) / tests))
    print("Wrote test runner, starting analysis")
    run_analysis(entry, analyses, coverage=True)
    print("Finished analysis, copying coverage")
    shutil.copy("/tmp/dynapyt_coverage/covered.json", "/Work/reports/")
