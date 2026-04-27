docker run -v $(pwd)/reports/:/Work/reports/ dylin_project $1 $2 > log_$1.txt  # $1 = index into project_repos.txt; $2 = nocov|cov; container writes /Work/reports
mv reports project_results/reports_$1  # Save this run’s output as reports_<index>
mkdir reports  # Empty dir for next docker bind mount
