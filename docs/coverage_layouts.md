# Coverage artifacts: expected file layouts

This document defines the expected on-disk layouts for **RQ4** (analysis coverage vs test coverage).
It is intended as a contract for `scripts/coverage_report.py` and as a reference when runs fail.

## Definitions

- **Project index**: the 1-based index of a benchmark project as listed in `scripts/projects.txt`.
- **Reports dir**: `project_results/reports_<index>/`
- **Testcov dir**: `project_testcovs/testcov_<index>/`

## DyLin analysis coverage (from `cov` runs)

When you run DyLin with coverage:

```bash
bash run_single_project.sh <index> cov
```

Expected output:

```
project_results/
  reports_<index>/
    timing.txt
    dynapyt_output/
      dynapyt_output-<session>/
        findings.csv
        output.json
    dynapyt_coverage/
      dynapyt_coverage-<session>/
        coverage.json
        coverage-<uuid>.json            # shard files (may exist)
        coverage-<uuid>.json            # shard files (may exist)
```

Notes:
- The **merged** analysis coverage file is `coverage.json`.
- The `coverage-<uuid>.json` files are intermediate shards and should not be used directly for RQ4.

## Test coverage (pytest-cov JSON)

When you run test coverage:

```bash
bash run_single_testcov.sh <index>
```

Expected output:

```
project_testcovs/
  testcov_<index>/
    cov.json
    timing.txt
```

Observed alternative layout (usually caused by reruns + `mv` into an existing directory):

```
project_testcovs/
  testcov_<index>/
    testcov/
      cov.json
      timing.txt
```

`scripts/coverage_report.py` supports both layouts, but the **preferred** layout is the non-nested one.

## Common failure modes

- **Missing `cov.json`**: the test coverage run failed before writing the JSON (dependency issues or pytest plugin conflicts).
- **Missing `coverage.json`**: the DyLin `cov` run did not complete for that index.
- **Nested output directories**: rerunning an index without cleaning the destination can nest directories (e.g., `reports_<i>/reports/` or `testcov_<i>/testcov/`).

