docker run --cpus="40" --memory="180g" -v /home/eghbalaz/DyLin/lint_reports/:/Work/lint_reports/ lint_project $1 > log_lint_$1.txt
mv lint_reports project_lints/lint_$1
mkdir lint_reports
