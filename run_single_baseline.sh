docker run -v $(pwd)/reports/:/Work/reports/ dylin_project_bl $1 > log_$1.txt
mv reports project_bl_results/reports_$1
mkdir reports
