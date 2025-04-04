docker run -v $(pwd)/testcov/:/Work/testcov/ testcov_project $1 > log_testcov_$1.txt
mv testcov project_testcovs/testcov_$1
mkdir testcov
