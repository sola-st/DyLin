README_MISMATCHES.txt
Track places where the repo’s instructions/results did not line up with the README or required extra setup.

Format:
- YYYY-MM-DD: Short title
  - Details / what happened

Entries from our setup/test runs so far.

- 2026-04-03: README micro-benchmark `requirements-tests.txt` fails on macOS
  - README suggests `pip install -r requirements-tests.txt`, but installation failed while building `nvidia-cublas-cu11` (CUDA/NVIDIA wheels are incompatible with this environment).
  - Workaround used: install `requirements.txt` + `pytest` (and `pip install -e .`) and run the micro-benchmark via `pytest tests/run_single_test.py`.

- 2026-04-03: README references missing `build_projects.sh`
  - README says `bash build_projects.sh`, but this repo only has `build_project.sh`.
  - Workaround used: `bash build_project.sh`.

- 2026-04-03: README typo `sumarize_findings.py`
  - README references `scripts/sumarize_findings.py`, but the actual script is `scripts/summarize_findings.py`.

- 2026-04-03: README “run all” script appears to cover only a small range
  - `run_all_no_cov.sh` exists, but in this repo it only loops over project indices `36..37`.

- 2026-04-03: pytest `-n/--timeout` flags need extra plugins
  - Attempting `pytest ... -n auto --timeout=...` initially errored due to unrecognized args until `pytest-xdist` and `pytest-timeout` were installed.

- 2026-04-03: Full local micro-benchmark has 3 failures (+ missing `tqdm`)
  - Running the full local micro-benchmark ended with 3 failures: `invalid_comparison`, `markings/leaked_data`, `ml/pytorch_gradient`.
  - For `ml/pytorch_gradient`, captured output shows `ModuleNotFoundError: No module named 'tqdm'` inside the instrumented program.

- 2026-04-03: IDE interpreter mismatch (false “filelock not installed”)
  - Cursor/pyright initially reported `filelock` missing due to using a different interpreter; setting Cursor to `./.venv/bin/python` fixed it.

- 2026-04-03: coverage scripts run only a tiny subset of projects
  - `run_all_no_cov.sh` runs only indices `36..37`.
  - `run_all_with_cov.sh` runs only indices `37..38` (and thus not the intended `1..37` GitHub projects).
  - `run_all_testcov.sh` runs only indices `36..37`.

- 2026-04-03: coverage report script used a fixed 1..49 loop
  - Updated to default to the number of projects in `scripts/projects.txt` (37); optional `--max_project` for partial runs.

- 2026-04-06: `coverage_report.py` looked for DyLin coverage at the wrong path and matched shard files
  - Original glob `reports_<i>/dynapyt_coverage-*/coverage*.json` does not match `reports_<i>/dynapyt_coverage/dynapyt_coverage-*/coverage.json`.
  - DynaPyt also writes multiple `coverage-<uuid>.json` files; the merged file is `coverage.json`. Fixed in-repo to use `.../dynapyt_coverage/dynapyt_coverage-*/coverage.json` and to resolve `testcov_<i>/testcov/cov.json` when present.

- 2026-04-03: `scripts/compare_static_dynamic_linters.py` looked for `results.txt` only
  - Linter runs produce `results_ruff.txt`, `results_pylint.txt`, and `results_mypy.txt`, so the script could match no files and print nothing.
  - Fixed in-repo to scan those filenames (and still allow `results.txt` if present).

- 2026-04-03: `run_single_linter.sh` did not create `project_lints/`
  - It runs `mv lint_reports project_lints/lint_$1` without `mkdir -p project_lints`, so `mv` fails if that directory is missing and results never land under `project_lints/`.
  - Fixed in-repo by adding `mkdir -p project_lints lint_reports` at the start of `run_single_linter.sh`. Re-run `bash run_all_linters.sh` after pulling that change (or create `project_lints` manually once).

- 2026-04-03: `scripts/summarize_findings.py` did not match Docker output layout
  - It looked for `timing.txt` at `reports_<i>/dynapyt_output/timing.txt`, but `analyze_repo.sh` places it at `reports_<i>/timing.txt`.
  - It expected per-checker `*report.json` files next to `findings.csv`, but current runs produce a single `output.json` (list of session blocks) in that directory.
  - Fixed in-repo by updating `summarize_findings.py` to resolve `timing.txt` correctly and read findings from `output.json`.

- 2026-04-06: `run_single_project.sh` / `run_single_testcov.sh` could fail to move outputs on fresh clones
  - Both scripts used `mv reports project_results/reports_$i` and `mv testcov project_testcovs/testcov_$i` without creating destination parents first.
  - Fixed in-repo by adding `mkdir -p project_results reports` and `mkdir -p project_testcovs testcov` at the start of the scripts.

- 2026-04-06: Test coverage (`testcov`) runs are not reproducible across all 37 projects due to dependency contamination
  - The `testcov_project` container installs each repo’s requirements into a shared environment; some projects break the environment for later ones (e.g., incompatible pytest/pluggy/pytest-cov versions).
  - Symptom: some `project_testcovs/testcov_<i>/cov.json` files are missing because pytest never successfully starts.

- 2026-04-06: Project-specific `testcov` failures observed (examples)
  - index 7 (`adversarial-robustness-toolbox`): `ValueError: numpy.dtype size changed...` (NumPy / scikit-learn binary incompatibility) prevented imports.
  - index 15 (`thefuck`): editable install/build issues (`pkg_resources` missing) and `ModuleNotFoundError` prevented tests from running.
  - index 23 (`click`): pytest tooling mismatch (e.g., `pytest-cov` requiring newer `pluggy`) led to `TypeError: HookimplMarker.__call__() got an unexpected keyword argument 'wrapper'`.
  - index 36 (`keras`): dependency resolution issues (e.g., `tensorflow-cpu~=2.17.0` not found, missing `jax`) prevented conftest imports.

- 2026-04-06: `project_testcovs` sometimes contains nested `testcov/` directories
  - If `mv` happens while `testcov_<i>` already exists, outputs can end up as `project_testcovs/testcov_<i>/testcov/cov.json` instead of `.../testcov_<i>/cov.json`.
  - Fixed in-repo by teaching `coverage_report.py` to also look for the nested `.../testcov/cov.json`, but the directory nesting is still confusing and should be avoided by cleaning the destination before reruns.

