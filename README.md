# DyLin
Dynamic Linter for Python

## Requirements
You need `docker`, `git`, and `python>=3.9` installed for running the experiments.
For Kaggle experiments, you need to have a Kaggle API key set in `kaggle.json` in the root directory.

## Checkers
The checkers are implemented in `src/analyses`.

## Evaluation
All experiments are self-contained, i.e. they download the required repositories, source code, and data.
Run the micro-benchmark:
```bash
pytest tests
```

Run DyLin on GitHub repositories:
```bash
bash build_projects.sh
bash run_all.sh
```
Results will be in `project_results`.

Run the GitHub project's test suites:
```bash
bash build_testcov.sh
bash run_all_testcov.sh
```
Results will be in `project_testcovs`.

Run Ruff on GitHub projects:
```bash
bash build_lint.sh
bash run_all_linters.sh
```
Results will be in `project_lints`.

Run DyLin on a Kaggle competition:
```bash
bash build_kaggle.sh <kaggle competition id: e.g. titanic>
bash run_kaggle.sh
```
Results will be in `kaggle_results`.
