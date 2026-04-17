docker run -v $(pwd)/lint_reports/:/Work/lint_reports/ lint_project $1 > log_lint_$1.txt  # $1 = project index; outputs results_*.txt under lint_reports
mv lint_reports project_lints/lint_$1  # Persist as lint_<index>
mkdir lint_reports  # Fresh mount point for next run
