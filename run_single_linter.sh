set -euo pipefail

# Optional: send all outputs under OUT_ROOT (absolute or relative to repo root).
# Example: OUT_ROOT="outputs/RQ3_linters" bash run_single_linter.sh 1
OUT_ROOT="${OUT_ROOT:-.}"
OUT_ROOT_ABS="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$OUT_ROOT")"

mkdir -p "$OUT_ROOT_ABS/project_lints" "$OUT_ROOT_ABS/lint_reports"

docker run -v "$OUT_ROOT_ABS/lint_reports/:/Work/lint_reports/" lint_project "$1" > "$OUT_ROOT_ABS/log_lint_$1.txt"
rm -rf "$OUT_ROOT_ABS/project_lints/lint_$1"
mv "$OUT_ROOT_ABS/lint_reports" "$OUT_ROOT_ABS/project_lints/lint_$1"
# mkdir -p "$OUT_ROOT_ABS/lint_reports"
