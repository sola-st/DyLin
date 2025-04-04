docker run -v $(pwd)/reports/:/Work/reports/ dylin_project $1 $2 > log_$1.txt
mv reports project_results/reports_$1
mkdir reports
