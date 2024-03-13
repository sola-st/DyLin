docker run --cpus="40" --memory="200g" -v /home/eghbalaz/DyLin/testcov/:/Work/testcov/ testcov_project $1 > log_testcov_$1.txt
mv testcov project_testcovs/testcov_$1
mkdir testcov
