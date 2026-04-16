# Root directory guide

This document explains the purpose and relationships of the files **tracked at the repository root** (what you get after a fresh `git clone`).

Notes:
- Many `log_*.txt`, `project_*`, `reports/`, `lint_reports/`, and `testcov/` paths you may see locally are **generated outputs** from running the scripts; they are typically not committed.
- Most large-scale experiments run in Docker. If you change a `Dockerfile.*`, you must rebuild the corresponding image before re-running the experiment (see “Build + run workflow” below).

## Build + run workflow (reproduction-oriented)

DyLin’s scripts generally follow this pattern:
- **Build an image**: `build_*.sh` (wraps `docker build ...`)
- **Run one project**: `run_single_*.sh` (wraps `docker run ...` and moves the mounted output directory)
- **Run many projects**: `run_all_*.sh` loops over indices and calls the corresponding `run_single_*.sh`

Important expectations/workarounds (reflecting current script behavior):
- `run_single_project.sh` mounts `./reports/` into the container and then moves `reports/` into `project_results/…`.
- `run_single_linter.sh` mounts `./lint_reports/` and then moves it into `project_lints/…`.
- `run_single_testcov.sh` mounts `./testcov/` and then moves it into `project_testcovs/…`.
- **First-run setup**: the scripts recreate `reports/`, `lint_reports/`, `testcov/` after each run, but they do not create the parent output directories (`project_results/`, `project_lints/`, `project_testcovs/`). Create them once before running, e.g.:

```bash
mkdir -p reports lint_reports testcov project_results project_lints project_testcovs
```

## Root-level files

### Core docs and metadata
- **`README.md`**: Main entry point; describes DyLin, basic requirements, and reproduction commands.
- **`INSTALL`**: Minimal install/run instructions (microbenchmark + Docker-based runs).
- **`LICENSE`**: Project license terms.
- **`STATUS`**: Artifact/badging status statement.

### Python packaging and dependencies
- **`pyproject.toml`**: Python project configuration (tooling and packaging metadata).
- **`requirements.txt`**: Base Python dependencies (used for local runs / development).
- **`requirements-tests.txt`**: Test dependencies for running `pytest` on the microbenchmark.
- **`test_requirements.txt`**: Additional test requirements (used by some benchmark/test flows).
- **`REQUIREMENTS`**: High-level system requirements (OS/tooling prerequisites).

### Dockerfiles (experiments run in containers)
- **`Dockerfile.project`**: Image for running DyLin on GitHub projects.
- **`Dockerfile.baseline_project`**: Image for baseline project runs (non-DyLin comparison).
- **`Dockerfile.linter`**: Image for running static linters on GitHub projects.
- **`Dockerfile.testcov`**: Image for running project test suites to collect test coverage.
- **`Dockerfile.kaggle`**: Image for Kaggle competition experiments.
- **`.dockerignore`**: Docker build context exclusions.

### Build scripts (build Docker images)
- **`build_project.sh`**: Builds `dylin_project` from `Dockerfile.project`.
- **`build_lint.sh`**: Builds `lint_project` from `Dockerfile.linter`.
- **`build_testcov.sh`**: Builds `testcov_project` from `Dockerfile.testcov`.
- **`build_kaggle.sh`**: Builds the Kaggle image from `Dockerfile.kaggle`.

### Run scripts (execute experiments)
- **`run_single_project.sh`**: Run DyLin on one GitHub project index (e.g. `nocov` vs `cov` mode).
- **`run_all_no_cov.sh`**: Loop over a configured index range and call `run_single_project.sh ... nocov`.
- **`run_all_with_cov.sh`**: Loop over a configured index range and call `run_single_project.sh ... cov`.
- **`run_single_linter.sh`**: Run static linters for one GitHub project index.
- **`run_all_linters.sh`**: Loop over indices and call `run_single_linter.sh`.
- **`run_single_testcov.sh`**: Run a project’s tests to collect test coverage for one index.
- **`run_all_testcov.sh`**: Loop over indices and call `run_single_testcov.sh`.
- **`run_single_baseline.sh`**: Run the baseline image for one GitHub project index.
- **`run_all_bl.sh`**: Loop over indices and call `run_single_baseline.sh`.
- **`run_kaggle.sh`**: Run the Kaggle workflow (requires `kaggle.json` in the repo root).
- **`clean_up_reports.sh`**: Utility to delete container-mounted output content under `project_results/` (use with care).

### Configuration / inputs (GitHub and Kaggle runs)
- **`project_repos.txt`**: List of GitHub repositories used in the project benchmark.
- **`dylin_config_project.txt`**: DyLin configuration for GitHub project runs.
- **`dylin_config_kaggle.txt`** / **`dylin_config_kaggle_temp.txt`**: DyLin configuration variants for Kaggle runs.

### Source, scripts, and tests (tracked directories)
- **`src/`**: DyLin implementation. Analyses/checkers live under `src/analyses/`.
- **`scripts/`**: Helper utilities for summarizing findings, comparing tools, and reporting coverage.
- **`tests/`**: Microbenchmark tests (run with `pytest tests`).
- **`test_projects/`**: Small projects/fixtures used by tests and experimentation.
- **`cli_tests/`**: Tests related to the CLI behavior.
- **`Supplementary_Material_FSE2025/`**: Supplementary artifact material supporting the paper (e.g., PDFs).

### Paper artifacts / computed summaries (tracked)
- **`paper_dylin.pdf`**: Paper PDF included with the artifact.
- **`mypy_findings.txt`**, **`ruff_warning_count.txt`**, **`res_loc.txt`**, **`loc_report.txt`**, **`line_count_res.txt`**: Precomputed summary files used for evaluation/analysis reporting.

