# DyLin CLI Demo Project

A minimal project to test the `dylin` CLI end-to-end.  
It contains deliberate bugs that should be caught by two DyLin checkers.

| File | Checker triggered | Code (DyLin issue) |
|------|------------------|--------------------|
| `main.py` | **SL-01 InPlaceSort** | `sorted()` called on a list but the return value is discarded |
| `main.py` | **PC-01 InvalidFunctionComparison** | A function object is compared to a number with `==` |

## Prerequisites

- Docker daemon running
- DyLin and the Docker Python package installed: `pip install -e . docker` (from the repo root)

## Running

From the **repo root**:

```bash
mkdir -p /tmp/dylin_cli_demo_output

dylin \
  --project-root cli_tests/cli_demo \
  --include      SL-01,PC-01 \
  --output-dir   /tmp/dylin_cli_demo_output \
  --coverage \
  -- python main.py
```

The CLI will:
1. Build a Docker image with DynaPyt installed.
2. Copy the project inside the container.
3. Instrument `main.py` with the selected checkers.
4. Run `python main.py` and save its output in `run_output.log`.
5. Store the merged coverage files in the same output directory.

Inspect `output.txt`, `output.json`, and `findings.csv` in the output directory for findings. See the [main README](../../README.md) for direct mode, custom analyses, and more options.
