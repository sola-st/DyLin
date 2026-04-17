# Root directory guide

This document explains the purpose, usage, and relationships of **every file and directory tracked at the repository root** (what you get after a fresh `git clone`). For a shorter onboarding path, see the **High-value** subsection below; the **complete list** satisfies the “every root item” requirement.

**Related:** the main overview and commands live in `README.md` (which links here).

---

## How root pieces fit together

- **Host tools:** `git`, `docker`, and `python>=3.9` are expected for the evaluation workflows (`README.md`, `REQUIREMENTS`, `INSTALL`).
- **Python deps:** `requirements.txt` (runtime/dev), `requirements-tests.txt` and `test_requirements.txt` (tests/benchmark harness), `pyproject.toml` (packaging and tool settings).
- **Docker:** Each `Dockerfile.*` defines one experiment image; `build_*.sh` wraps `docker build`. **After you change a Dockerfile or copied deps, rebuild** before re-running.
- **Benchmark inputs:** `project_repos.txt` lists clone URLs (one per line; **1-based index** selects a project). `dylin_config_project.txt` and `dylin_config_kaggle*.txt` list **which DyLin analysis classes** run (one qualified class name per line; some lines include `;config=...` for `ObjectMarkingAnalysis`).
- **Run wrappers:** `run_single_*.sh` bind-mount an output folder, run the container, move output to `project_*` folders, and recreate the mount directory. Create parent dirs once, e.g. `mkdir -p reports lint_reports testcov project_results project_lints project_testcovs` (and `kaggle_results` for Kaggle).

---

## Expected outputs (not usually committed)

Running the scripts creates directories and logs such as `project_results/`, `project_lints/`, `project_testcovs/`, `log_*.txt`, and working dirs `reports/`, `lint_reports/`, `testcov/`. These are **generated**; they are not part of the tracked root list unless you add them to Git.

Typical artifacts per workflow (names may vary by DynaPyt session id):

| Workflow | Where it lands | Notable files |
|----------|----------------|---------------|
| DyLin on repos | `project_results/reports_<i>/` | `dynapyt_output-*/findings.csv`, `output.json`; timing often in `timing.txt` |
| Static linters | `project_lints/lint_<i>/` | `results_ruff.txt`, `results_pylint.txt`, `results_mypy.txt` |
| Test coverage | `project_testcovs/testcov_<i>/` | pytest-cov output (e.g. `cov.json`) |
| Kaggle | `kaggle_results/` (mounted) | Competition-specific outputs from `scripts/analyze_kaggle.sh` |

---

## High-value root items (start here)

| Item | Role |
|------|------|
| `README.md` | Main entry: concept, requirements, RQ1–RQ4 commands |
| `INSTALL` | Minimal steps for microbenchmark and Docker runs |
| `REQUIREMENTS` | Supported OS and required tools (one line) |
| `Dockerfile.project` + `build_project.sh` + `run_single_project.sh` | GitHub-project DyLin pipeline |
| `Dockerfile.linter` + `build_lint.sh` + `run_single_linter.sh` | Static comparison (Ruff, Pylint, Mypy) |
| `Dockerfile.testcov` + `build_testcov.sh` + `run_single_testcov.sh` | Test-suite coverage without DyLin |
| `project_repos.txt` | Benchmark repository list (index = line number) |
| `dylin_config_project.txt` | Which analyses run on GitHub projects |

---

## Complete list of tracked root-level files and directories

Alphabetical. **Every** tracked root path appears exactly once with a short purpose line.

| Path | Purpose |
|------|---------|
| `.dockerignore` | Excludes files from the Docker build context (smaller/faster builds, fewer secrets). |
| `.github/` | GitHub Actions workflows and other GitHub automation for this repository. |
| `.gitignore` | Tells Git which generated or machine-local paths not to commit. |
| `Dockerfile.baseline_project` | Builds image for **baseline** (non-DyLin) benchmark runs; entrypoint runs `scripts/baseline_repo.py`. |
| `Dockerfile.kaggle` | Builds Kaggle experiment image from `gcr.io/kaggle-images/python`; competition id is a **build-arg**; expects `kaggle.json` in build context. |
| `Dockerfile.linter` | Builds image that runs static linters (`lint_repo.sh`) on benchmark repos. |
| `Dockerfile.project` | Builds image that runs DyLin on benchmark GitHub repos (`analyze_repo.sh`). |
| `Dockerfile.testcov` | Builds image that runs each repo’s tests with pytest-cov (`testcov_repo.sh`). |
| `INSTALL` | Short install/run instructions for the microbenchmark and container-based workflows. |
| `LICENSE` | Legal license for the project. |
| `README.md` | Primary documentation: DyLin overview, requirements, and evaluation commands. |
| `REQUIREMENTS` | States tested platform and required external tools (`git`, `docker`, Python version). |
| `STATUS` | Artifact / badge narrative for the paper’s availability and reusability claims. |
| `Supplementary_Material_FSE2025/` | Supplementary PDFs and materials for the FSE 2025 artifact. |
| `build_kaggle.sh` | Runs `docker build` for the Kaggle image; first argument is the Kaggle competition id. |
| `build_lint.sh` | Builds the `lint_project` image from `Dockerfile.linter`. |
| `build_project.sh` | Builds the `dylin_project` image from `Dockerfile.project`. |
| `build_testcov.sh` | Builds the `testcov_project` image from `Dockerfile.testcov`. |
| `clean_up_reports.sh` | **Destructive:** clears contents of persisted `project_results/` via a throwaway Ubuntu container. |
| `cli_tests/` | Tests focused on the DyLin **CLI** (separate from `tests/`). |
| `docs/` | Extra documentation (including this file). |
| `dylin_config_kaggle.txt` | List of analysis classes used for **Kaggle** runs (paths inside container use `/Work/src/...` where applicable). |
| `dylin_config_kaggle_temp.txt` | Alternate Kaggle analysis list / layout variant for experiments. |
| `dylin_config_project.txt` | List of analysis classes for **GitHub-project** runs (paths use `/Work/DyLin/...` for marking configs). |
| `line_count_res.txt` | Precomputed line-count summary used in paper/artifact reporting. |
| `loc_report.txt` | Precomputed lines-of-code report data for evaluation tables. |
| `mypy_findings.txt` | Precomputed mypy-oriented summary for static-vs-dynamic comparisons. |
| `paper_dylin.pdf` | PDF copy of the DyLin paper bundled with the repository. |
| `project_repos.txt` | Ordered Git repository URLs for the benchmark; **project index = line number** (1-based) for `run_single_*.sh`. |
| `pyproject.toml` | Python package metadata, build system, and tool configuration (e.g. pytest). |
| `requirements-tests.txt` | Pip dependencies to run `pytest` on the microbenchmark and test suite. |
| `requirements.txt` | Pip dependencies for local development and for Docker image installs. |
| `res_loc.txt` | Precomputed result/location summary data used in evaluation. |
| `ruff_warning_count.txt` | Precomputed Ruff warning counts for RQ3-style analysis. |
| `run_all_bl.sh` | Loops over indices and calls `run_single_baseline.sh` (baseline batch). |
| `run_all_linters.sh` | Loops over indices and calls `run_single_linter.sh` (static linter batch). |
| `run_all_no_cov.sh` | Loops over indices and calls `run_single_project.sh` with `nocov` (DyLin without analysis coverage). |
| `run_all_testcov.sh` | Loops over indices and calls `run_single_testcov.sh` (test coverage batch). |
| `run_all_with_cov.sh` | Loops over indices and calls `run_single_project.sh` with `cov` (DyLin with analysis coverage). |
| `run_kaggle.sh` | Runs the built `dylin_kaggle` image; mounts `./kaggle_results` to `/Work/results`. |
| `run_single_baseline.sh` | Runs one baseline project index; image tag must match your build (e.g. `dylin_project_bl`). |
| `run_single_linter.sh` | Runs static linters for one index; moves `lint_reports/` → `project_lints/lint_<i>/`. |
| `run_single_project.sh` | Runs DyLin for one index and mode (`nocov` or `cov`); moves `reports/` → `project_results/reports_<i>/`. |
| `run_single_testcov.sh` | Runs pytest-cov for one index; moves `testcov/` → `project_testcovs/testcov_<i>/`. |
| `scripts/` | Shell and Python utilities: clone/analyze repos, lint, test coverage, summarize findings, coverage reports. |
| `src/` | DyLin package source; analyses live under `src/dylin/analyses` (see also `README.md` “Checkers”). |
| `test_projects/` | Small example projects used as fixtures in tests and benchmarks. |
| `test_requirements.txt` | Additional pip requirements for specific test or benchmark setups. |
| `tests/` | Pytest suite for the microbenchmark and core behavior. |

---

## Review checklist (backlog alignment)

- [x] All tracked root-level paths listed (table above).
- [x] Each path has a brief purpose description.
- [x] Configs (`dylin_config_*.txt`, `project_repos.txt`) and dependencies (`requirements*.txt`, `pyproject.toml`, Dockerfiles) explained in context.
- [x] Linked from `README.md` as the dedicated doc for root layout.

**Remaining task for you:** commit this branch and push to the remote when your team is ready (`git push -u origin <branch>`), then open a PR into `main` if required by the course.
