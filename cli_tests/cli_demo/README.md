# DyLin CLI Demo Project

A minimal project to test the `dylin` CLI end-to-end.  
It contains deliberate bugs that should be caught by two DyLin checkers.

| File | Checker triggered | Code (DyLin issue) |
|------|------------------|--------------------|
| `main.py:12` | **SL-01 InPlaceSort** | `sorted()` called on a list but the return value is discarded |
| `main.py:19` | **PC-01 InvalidFunctionComparison** | A function object is compared to `None` with `==` |

## Prerequisites

- Docker daemon running
- DyLin installed in the current Python environment: `pip install -e .` (from the repo root)

## Running

From the **repo root**:

```bash
mkdir -p /tmp/dylin_cli_demo_output

dylin \
  --project-root test_projects/cli_demo \
  --analysis     test_projects/cli_demo/analysis.txt \
  --output-dir   /tmp/dylin_cli_demo_output \
  -- python main.py
```

The CLI will:
1. Build a Docker image with DynaPyt installed.
2. Copy the project inside the container.
3. Instrument `main.py` with the selected checkers.
4. Run `python main.py` and stream output to your terminal.

Findings from both checkers should appear in the streamed output.
