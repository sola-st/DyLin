# DyLin
Dynamic Linter for Python

## Checkers
The checkers are implemented in `src/analyses`.

## Evaluation
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
