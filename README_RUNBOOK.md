# DyLin artifact runbook (commands & outputs)

This file supplements the upstream `README.md` with **exact commands**, **where outputs land**, and **fixes** that apply to this repository. For known mismatches with the original README, see `README_MISMATCHES.txt`.

---

## Prerequisites

| Need | Used for |
|------|----------|
| Docker Desktop (running) | GitHub projects, linters, test coverage, Kaggle |
| `git`, Python ≥ 3.9 | Local venv / tooling |
| Enough disk & RAM | Images + clones + logs are large |

**Optional:** `kaggle.json` in the **repo root** (see [Kaggle](#kaggle-optional)) — not required for GitHub / micro-benchmark workflows.

---

## Local micro-benchmark (pytest)

**Purpose:** Fast sanity check of DyLin checkers (RQ1-style “micro-bench” in the paper).

```bash
cd /path/to/DyLin
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
pip install pytest pytest-xdist pytest-timeout   # if you use -n / --timeout
pytest tests/run_single_test.py
```

| Output | Location |
|--------|----------|
| Per-test temp artifacts | Under `tests/<case>/` during the run (cleaned up after pass) |
| Pytest cache | `.pytest_cache/` |

**Note (macOS / Apple Silicon):** `requirements-tests.txt` from the upstream README may try to install NVIDIA CUDA wheels and **fail**. Using `requirements.txt` + `pip install -e .` as above is the usual workaround.

---

## Docker image builds (one-time per Dockerfile change)

Build only what you need:

| Command | Docker image | When |
|---------|--------------|------|
| `bash build_project.sh` | `dylin_project` | GitHub repos: DyLin `nocov` / `cov` |
| `bash build_lint.sh` | `lint_project` | Ruff / Pylint / Mypy on repos |
| `bash build_testcov.sh` | `testcov_project` | Plain pytest coverage (no DyLin) |
| `bash build_kaggle.sh <competition_id>` | `dylin_kaggle` | Kaggle pipeline only |

---

## GitHub projects — DyLin (`run_single_project.sh`)

**Upstream README says `build_projects.sh` — in this repo the script is `build_project.sh` (singular).**

### 1) Build the analysis image

```bash
bash build_project.sh
```

### 2) Run one project or all 37

Project index `i` is **1-based** and matches line `i` in `scripts/projects.txt`.

```bash
mkdir -p project_results reports

# nocov: runs the project test suite multiple times (see scripts/analyze_repo.py); slow.
bash run_single_project.sh <i> nocov

# cov: one pytest pass + DyLin analysis coverage; use for RQ4-style coverage.
bash run_single_project.sh <i> cov
```

**All 37 (example):**

```bash
for i in {1..37}; do
  bash run_single_project.sh "$i" nocov   # or: cov
done
```

| Output | Location |
|--------|----------|
| Docker log (host) | `log_<i>.txt` (repo root) |
| DyLin + timing | `project_results/reports_<i>/` |
| Findings | `project_results/reports_<i>/dynapyt_output/dynapyt_output-1234-abcd/findings.csv` |
| Full JSON sessions | `.../dynapyt_output-1234-abcd/output.json` |
| Timing | `project_results/reports_<i>/timing.txt` |
| **Only for `cov`:** analysis coverage JSON | `project_results/reports_<i>/dynapyt_coverage/dynapyt_coverage-1234-abcd/coverage*.json` |

**Backing up before a second mode (`nocov` vs `cov`):** both modes use the same `project_results/reports_<i>` names. **Rename or copy** the folder first, e.g. `mv project_results project_results_nocov_rq1`, then recreate `project_results` for the next run. Otherwise `mv` may **nest** folders instead of replacing them.

---

## Summarize DyLin findings (one text file)

```bash
source .venv/bin/activate   # if using venv
python scripts/summarize_findings.py --results project_results
```

| Output | Location |
|--------|----------|
| Aggregated DyLin lines | `project_results/DyLin_findings.txt` |

If you summarized from a **renamed** tree, pass that path instead of `project_results`:

```bash
python scripts/summarize_findings.py --results project_results_nocov_rq1
```

---

## Static linters (Ruff / Pylint / Mypy)

```bash
bash build_lint.sh
bash run_all_linters.sh
```

`run_single_linter.sh` creates **`project_lints`** before moving outputs (required on a fresh clone).

| Output | Location |
|--------|----------|
| Docker log | `log_lint_<i>.txt` (repo root) |
| Ruff / Pylint / Mypy | `project_lints/lint_<i>/results_ruff.txt`, `results_pylint.txt`, `results_mypy.txt` |

---

## Compare static vs DyLin (`file:line` overlap)

```bash
python scripts/compare_static_dynamic_linters.py \
  --static_dir project_lints \
  --dynamic project_results/DyLin_findings.txt
```

| Output | Location |
|--------|----------|
| Matches (printed to terminal) | stdout only — redirect if you want a file: `> static_dynamic_overlap.txt` |

This repository’s script expects `results_ruff.txt` / `results_pylint.txt` / `results_mypy.txt` (not only `results.txt`).

---

## Test coverage only (no DyLin) — RQ4 input

```bash
bash build_testcov.sh
mkdir -p project_testcovs testcov

for i in {1..37}; do
  bash run_single_testcov.sh "$i"
done
```

| Output | Location |
|--------|----------|
| Docker log | `log_testcov_<i>.txt` (repo root) |
| Pytest coverage JSON | **`project_testcovs/testcov_<i>/cov.json`** |

**Expected layout for the next step:** `cov.json` must live **directly** under `testcov_<i>/`. If a failed `mv` left a nested `project_testcovs/testcov_<i>/testcov/cov.json`, move files up or rerun with a clean `testcov_<i>` directory.

---

## RQ4 — Analysis coverage vs test coverage (CSV)

Requires:

- DyLin **`cov`** outputs under `project_results/reports_<i>/` with `dynapyt_coverage-*/coverage*.json`
- Test coverage `project_testcovs/testcov_<i>/cov.json` for the same indices

```bash
rm -f coverage_comparison.csv   # script appends; remove to avoid duplicate rows
python scripts/coverage_report.py coverage_comparison \
  --analysis_dir project_results \
  --test_dir project_testcovs
```

| Output | Location |
|--------|----------|
| Per-project comparison rows | **`coverage_comparison.csv`** (repo root; often gitignored) |

The script loops indices **`1..N`**, where **`N` defaults to the number of projects in `scripts/projects.txt`** (37). Override with `--max_project=3` for a smoke test. Missing DyLin or test `cov.json` is skipped with a clearer message. Use `--strict=true` to fail the command if any index is missing coverage inputs.

For the exact on-disk layout the scripts expect/handle, see `docs/coverage_layouts.md`.

**Optional:** `python scripts/summarize_coverage.py` looks for `cov_comp_*.json` files (not produced by default here) — only relevant if you generate those JSONs elsewhere.

---

## Kaggle (optional)

Needs **`kaggle.json`** in the **repo root** (copied into the image as `/Work/.kaggle/`).

```bash
bash build_kaggle.sh <competition_id>
bash run_kaggle.sh
```

| Output | Location |
|--------|----------|
| Results | `kaggle_results/` (mounted from the container) |

---

## Quick reference — where everything lands

| Workflow | Main output directory / files |
|----------|-------------------------------|
| DyLin GitHub `nocov` / `cov` | `project_results/reports_<i>/`, `log_<i>.txt` |
| DyLin summary | `project_results/DyLin_findings.txt` (or under your renamed folder) |
| Static linters | `project_lints/lint_<i>/results_*.txt`, `log_lint_<i>.txt` |
| Test coverage only | `project_testcovs/testcov_<i>/cov.json`, `log_testcov_<i>.txt` |
| RQ4 CSV | `coverage_comparison.csv` |
| Kaggle | `kaggle_results/` |

---

## Paper mapping (FSE DyLin)

| Paper topic | Typical artifact path |
|-------------|------------------------|
| RQ1 effectiveness (GitHub) | `project_results` + `DyLin_findings.txt` |
| RQ3 vs static tools | `project_lints` + `compare_static_dynamic_linters.py` |
| RQ4 analysis coverage | `project_results` (`cov`) + `project_testcovs` + `coverage_comparison.csv` |
| Micro-benchmark | `pytest tests/run_single_test.py` |

---

*Generated for this checkout; keep `README.md` as upstream reference and use this runbook for day-to-day reproduction.*
