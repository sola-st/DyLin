set -euo pipefail

# Optional: send all outputs under OUT_ROOT (absolute or relative to repo root).
# Example: OUT_ROOT="outputs/RQ4_testcov" bash run_single_testcov.sh 1
OUT_ROOT="${OUT_ROOT:-.}"
OUT_ROOT_ABS="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$OUT_ROOT")"

mkdir -p "$OUT_ROOT_ABS/project_testcovs" "$OUT_ROOT_ABS/testcov"

docker run -v "$OUT_ROOT_ABS/testcov/:/Work/testcov/" testcov_project "$1" > "$OUT_ROOT_ABS/log_testcov_$1.txt"
rm -rf "$OUT_ROOT_ABS/project_testcovs/testcov_$1"
mv "$OUT_ROOT_ABS/testcov" "$OUT_ROOT_ABS/project_testcovs/testcov_$1"
# mkdir -p "$OUT_ROOT_ABS/testcov"
