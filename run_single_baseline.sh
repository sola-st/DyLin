docker run --cpus="20" --memory="150g" -v /home/eghbalaz/DyLin/reports/:/Work/reports/ dylin_project_bl $1 > log_$1.txt
mv reports project_bl_results/reports_$1
mkdir reports
