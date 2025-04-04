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

Run the GitHub project's test suites without DyLin to get test suite coverage (used in RQ4):
```bash
bash build_testcov.sh
bash run_all_testcov.sh
```
Results will be in `project_testcovs`.

Run static linters on GitHub projects (RQ3):
```bash
bash build_lint.sh
bash run_all_linters.sh
```
Results will be in `project_lints`.

Run DyLin on a Kaggle competition (RQ1):
```bash
bash build_kaggle.sh <kaggle competition id: e.g. titanic>
bash run_kaggle.sh
```
Results will be in `kaggle_results`.
