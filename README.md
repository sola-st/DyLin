# DyLin
A Dynamic Linter for Python!  
> **What is a dynamic linter?**  
> Popular linters like pylint, ruff, etc. are static analyzers that check the source code for common mistakes and programming anti-patterns. However, because of the dynamic nature of Python, there are some common mistakes and anti-patterns that these linters cannot detect. A dynamic linter analyzes the code during execution and can find such anti-patterns.

## Run DyLin on your GitHub workflow
It is now possible to run DyLin directly on GitHub workflows. You just need to add 5 lines to your workflow file.
Checkout [this repository](https://github.com/AryazE/auto-dylin/) for more details.

## Installation

DyLin requires Python 3.10 or newer and Git. From a checkout, install it in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e . docker
```

The `docker` Python package is required by the CLI, including direct runs. The default runner also requires Docker, a running Docker daemon, and network access to pull the latest published image. The image's DyLin version may differ from your local checkout.

The container image is published at `ghcr.io/sola-st/dylin:latest` on pushes to `main`; each image is also tagged with its commit SHA. It includes Python 3.13, Git, GCC, pytest, and DyLin. To build it from a checkout and run it:

```bash
docker build -t dylin .
docker run --rm -it dylin bash
```

Inside the container, use `dylin --direct` to analyze a project mounted into the container. The default Docker runner requires access to a Docker daemon.

## Analyze a project

Run a command that exercises the code you want to inspect. DyLin detects behavior reached during that execution, so a test suite or representative workload is useful.

```bash
dylin --project-root cli_tests/cli_demo --output-dir /tmp/dylin-results \
  --include SL-01,PC-01 -- python main.py
```

Options must appear before `--`; the remaining arguments form a shell command executed from the project root. By default, DyLin copies the project into a container, instruments the copy, and runs the command there. Use `--setup 'pip install -r requirements.txt'` to install the target project's dependencies before instrumentation. The output directory is created on the host; use a fresh directory outside the target project for each run.

To run locally instead, install the target project's dependencies in the same environment and use:

```bash
dylin --direct --project-root /path/to/project --output-dir /tmp/dylin-results \
  --include SL-01 -- pytest tests
```

Direct mode instruments files in the project directory itself. Run it on a disposable copy or a clean checkout that you can restore. It uses Bash and writes a fixed DynaPyt session file in `/tmp`, so concurrent runs can interfere with one another.

### Select checkers and inspect results

By default `--include all` enables the registered checkers. Use comma-separated issue codes or checker names with `--include` and `--exclude`, for example `--include SL-01,PC-01` or `--exclude InPlaceSort`. The available codes and names are listed in [select_checkers.py](src/dylin/select_checkers.py). Some codes share an analysis, so selecting one can also produce related findings.

The output directory can contain `findings.csv` (counts by analysis), individual report JSON files, `output.json` (merged results), and `output.txt` (formatted findings). Instrumentation and post-processing logs are written to `dynapyt_instrumentation.log` and `dynapyt_post_run.log`. By default target output goes to `run_output.log`; `--quiet` instead leaves it on the runner's output stream. Add `--coverage` to collect analysis coverage, `--timeout 600` to limit runtime, or `--memory-limit 2g` to limit container memory. The memory limit does not apply in direct mode. Inspect the logs as well as the printed status when diagnosing a run; the CLI does not reliably propagate the target command's exit status.

### Custom analyses

Pass `--analysis path/to/analyses.txt` to replace checker selection with a DynaPyt analysis list, one fully qualified analysis class per line. See [the example list](cli_tests/analyses/dylin_analyses.txt). A custom Python file can also be passed directly:

```bash
dylin --direct --project-root cli_tests/cli_demo --output-dir /tmp/dylin-custom \
  --analysis cli_tests/analyses/custom_analysis.py -- python main.py
```

The file must contain a class that directly subclasses `BaseDyLinAnalysis`; the CLI selects the first such class it finds. See [custom_analysis.py](cli_tests/analyses/custom_analysis.py) for a hook and `add_finding` example. The analysis directory must be writable: the CLI writes `final_analysis.txt` there and, for a Python analysis, writes a sibling `.txt` file (overwriting one if it exists). In Docker mode, custom analysis files in that directory are mounted into the container.

## Experiments

The evaluation scripts require Docker, Git, and Python. For Kaggle experiments, put a Kaggle API key in `kaggle.json` in the root directory:
```json
{"username": "your username", "key": "your API key"}
```
## Checkers
The checkers are implemented in `src/dylin/analyses`.

## Evaluation
All experiments (except the micro-benchmark run) run inside Docker containers.
The scripts provided are self-contained, i.e. they build the required Docker container, and download the required repositories, source code, and data.

### RQ1: Effectiveness
Run DyLin on the micro-benchmark:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements-tests.txt
pytest tests
```

Run DyLin on the GitHub repositories:
```bash
bash build_project.sh
bash run_all_no_cov.sh
```
The supplied `run_all_no_cov.sh` currently runs indices 36 and 37 from `scripts/projects.txt`; adjust the loop to cover other projects. Likewise, `run_all_with_cov.sh` currently runs indices 37 and 38.
Results will be in `project_results`.
For each repository a directory is created with one sub-directory and a text file.
The sub-directory is named `dynapyt_output-<unique id of the run>`, with the following contents:
- A `findings.csv` file, which summarizes the findings.
- An `output.json` file, which contains the details of the findings.
The text file contains the name of the project, the instrumentation duration in seconds, and the analysis time in seconds.

You can pretty print the findings in a format similar to static linters by running:
```bash
python scripts/summarize_findings.py --results <path to the subdirectory in project_results>
```
This will generate a text file with the format
```
<file>:<line>:<column>: <issue code> <issue message>
```

Run DyLin on a Kaggle competition:
```bash
bash build_kaggle.sh <kaggle competition id: e.g. titanic>
bash run_kaggle.sh
```
Results will be in `kaggle_results`.
For each competition a directory is created with 3 subdirectories:
- `coverage`, which contains analysis coverage information in a json file.
- `submissions`, which contains the submissions analyzed by DyLin.
- `table`, which contains the findings in a json file.

### RQ2: Severity of Detected Issues
The submitted pull requests and issues are available in `Supplementary_Material_FSE2025/DyLin Issues - *.pdf`

### RQ3: Comparison with Existing Tools
Run static linters on GitHub projects:
```bash
bash build_lint.sh
bash run_all_linters.sh
```
Results will be in `project_lints`.
For each repository a directory is created with 3 files `results_ruff.txt`, `results_pylint.txt`, and `results_mypy.txt`.
To match the warning lines from the static linters to DyLin warnings, run:
```bash
python scripts/compare_static_dynamic_linters.py --static_dir <path to the directory containing the static linter results> --dynamic <path to the text file containing DyLin's findings>
```
This will output all lines that both approaches have warned about.

### RQ4: Analysis Coverage
Run DyLin with analysis coverage on:
```bash
bash build_project.sh
bash run_all_with_cov.sh
```
Run the GitHub project's test suites without DyLin to get test suite coverage:
```bash
bash build_testcov.sh
bash run_all_testcov.sh
```
Results will be in `project_testcovs`.
For each repository a directory is created with a json file containing the detailed test coverage data.

To calculate the ratio of analysis coverage to test coverage you can run
```bash
python scripts/coverage_report.py coverage_comparison --analysis_dir <path to the subdirectory in project_results> --test_dir <path to the subdirectory in project_testcovs>
```
This generates a csv file with a summary of analysis and test coverage similar to `Supplementary_Material_FSE2025/DyLin - FSE 2025 Artifact.pdf` page 1.
