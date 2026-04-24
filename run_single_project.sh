set -euo pipefail

# Optional: send all outputs under OUT_ROOT (absolute or relative to repo root).
# Example: OUT_ROOT="outputs/RQ4_cov" bash run_single_project.sh 1 cov
OUT_ROOT="${OUT_ROOT:-.}"
OUT_ROOT_ABS="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$OUT_ROOT")"

mkdir -p "$OUT_ROOT_ABS/project_results" "$OUT_ROOT_ABS/reports"

docker run -v "$OUT_ROOT_ABS/reports/:/Work/reports/" dylin_project "$1" "$2" > "$OUT_ROOT_ABS/log_$1.txt"
rm -rf "$OUT_ROOT_ABS/project_results/reports_$1"
mv "$OUT_ROOT_ABS/reports" "$OUT_ROOT_ABS/project_results/reports_$1"
mkdir -p "$OUT_ROOT_ABS/reports"
