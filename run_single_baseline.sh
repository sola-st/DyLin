docker run -v $(pwd)/reports/:/Work/reports/ dylin_project_bl $1 > log_$1.txt  # Baseline (non-DyLin) run; image name must match docker tag you built
mv reports project_bl_results/reports_$1  # Output tree parallel to project_results/
mkdir reports
