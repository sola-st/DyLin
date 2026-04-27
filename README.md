# DyLin
A Dynamic Linter for Python!  
> **What is a dynamic linter?**  
> Popular linters like pylint, ruff, etc. are static analyzers that check the source code for common mistakes and programming anti-patterns. However, because of the dynamic nature of Python, there are some common mistakes and anti-patterns that these linters cannot detect. A dynamic linter analyzes the code during execution and can find such anti-patterns.

## Run DyLin on your GitHub workflow
It is now possible to run DyLin directly on GitHub workflows. You just need to add 5 lines to your workflow file.
Checkout [this repository](https://github.com/AryazE/auto-dylin/) for more details.

## Requirements
You need `docker`, `git`, and `python>=3.9` installed for running the experiments.
For Kaggle experiments, you need to have a Kaggle API key set in `kaggle.json` in the root directory with the following format:
```json
{"username": "your username", "key": "your API key"}
```
To install requirements for a local (not in a container) run:
```bash
pip install -r requirements.txt
```

## Checkers
The checkers are implemented in `src/dylin/analyses`.

## Evaluation
All experiments (except the micro-benchmark run) run inside Docker containers.
The scripts provided are self-contained, i.e. they build the required Docker container, download the required repositories, and write outputs under the repository directory.

### Quick start: build images (one-time per Dockerfile change)

Build only what you need:

```bash
bash build_project.sh   # dylin_project
bash build_lint.sh      # lint_project
bash build_testcov.sh   # testcov_project
# Optional Kaggle:
# bash build_kaggle.sh <competition_id>
```

### Output layout (what the scripts produce)

- **DyLin GitHub runs**: `project_results/reports_<i>/`
  - `timing.txt`
  - `dynapyt_output/dynapyt_output-<session>/findings.csv`
  - `dynapyt_output/dynapyt_output-<session>/output.json`
  - (only for `cov`) `dynapyt_coverage/dynapyt_coverage-<session>/coverage.json`
- **Static linters**: `project_lints/lint_<i>/results_ruff.txt`, `results_pylint.txt`, `results_mypy.txt`
- **Test coverage (pytest-cov)**: `project_testcovs/testcov_<i>/cov.json`

The **project index `i` is 1-based** and matches line `i` in `scripts/projects.txt`.

### RQ1: Effectiveness
#### RQ1a: Micro-benchmark (local, pytest)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
pip install pytest pytest-xdist pytest-timeout
pytest tests/run_single_test.py
```

#### RQ1b: GitHub benchmark (DyLin `nocov`)

Build the analysis image:

```bash
bash build_project.sh
mkdir -p project_results reports
```

Run one project:

```bash
bash run_single_project.sh <i> nocov
```

Run all 37:

```bash
for i in {1..37}; do
  bash run_single_project.sh "$i" nocov
done
```

Summarize DyLin findings into one text file (used for RQ3 comparisons):

```bash
source .venv/bin/activate   # if using venv
python scripts/summarize_findings.py --results project_results
```

This creates `project_results/DyLin_findings.txt` formatted like:

`<file>:<line>:<column>: <issue code> <issue message>`

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
python scripts/compare_static_dynamic_linters.py \
  --static_dir project_lints \
  --dynamic project_results/DyLin_findings.txt
```
This will output all lines that both approaches have warned about.

### RQ4: Analysis Coverage
Run DyLin with analysis coverage on:
```bash
bash build_project.sh
mkdir -p project_results reports
bash run_single_project.sh <i> cov
```
Run the GitHub project's test suites without DyLin to get test suite coverage:
```bash
bash build_testcov.sh
mkdir -p project_testcovs testcov
bash run_single_testcov.sh <i>
```
Results will be in `project_testcovs`.
For each repository a directory is created with a json file containing the detailed test coverage data.

To calculate the ratio of analysis coverage to test coverage you can run
```bash
rm -f coverage_comparison.csv   # script appends; remove to avoid duplicate rows
python scripts/coverage_report.py coverage_comparison \
  --analysis_dir project_results \
  --test_dir project_testcovs
```
This generates a csv file with a summary of analysis and test coverage similar to `Supplementary_Material_FSE2025/DyLin - FSE 2025 Artifact.pdf` page 1.

Notes:

- `scripts/coverage_report.py` supports `--max_project=3` for a smoke test and `--strict=true` to fail if any index is missing coverage inputs.
- If your run outputs are stored under a different root (for example using `OUT_ROOT="outputs/RQ4_cov"`), pass those paths instead of `project_results` / `project_testcovs`.
