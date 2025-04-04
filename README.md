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
The checkers are implemented in `src/analyses`.

## Evaluation
All experiments are self-contained, i.e. they download the required repositories, source code, and data.
Run the micro-benchmark (RQ1):
```bash
pip install -r requirements-tests.txt
pytest tests
```

Run DyLin on GitHub repositories (RQ1, RQ4, and RQ5):
```bash
bash build_projects.sh
bash run_all_with_cov.sh # to collect analysis coverage
bash run_all_no_cov.sh # no analysis coverage
```
Results will be in `project_results`.
For each repository a directory is created with two sub-directories and a text file.
The first sub-directory is named `dynapyt_coverage-<unique id of the run>`, which contains a json file with detailed analysis coverage data.
The second sub-directory is named `dynapyt_output-<unique id of the run>`, with the following contents:
- A `findings.csv` file, which summarizes the findings.
- An `output.json` file, which contains the details of the findings.
The text file contains the name of the project, the instrumentation duration in seconds, and the analysis time in seconds.

You can pretty print the findings in a format similar to static linters by running:
```bash
python scripts/sumarize_findings.py --results <path to the subdirectory in project_results>
```
This will generate a text file with the format
```
<file>:<line>:<column>: <issue code> <issue message>
```

Run the GitHub project's test suites without DyLin to get test suite coverage (used in RQ4):
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

Run static linters on GitHub projects (RQ3):
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

Run DyLin on a Kaggle competition (RQ1):
```bash
bash build_kaggle.sh <kaggle competition id: e.g. titanic>
bash run_kaggle.sh
```
Results will be in `kaggle_results`.
For each competition a directory is created with 3 subdirectories:
- `coverage`, which contains analysis coverage information in a json file.
- `submissions`, which contains the submissions analyzed by DyLin.
- `table`, which contains the findings in a json file.
