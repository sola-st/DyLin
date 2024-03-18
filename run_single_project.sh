docker run --cpus="40" --memory="180g" -v /home/eghbalaz/DyLin/reports/:/Work/reports/ dylin_project $1 > log_$1.txt
mv reports project_results/reports_$1
mkdir reports
