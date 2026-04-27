docker run -v $(pwd)/testcov/:/Work/testcov/ testcov_project $1 > log_testcov_$1.txt  # $1 = project index; pytest-cov output → testcov/
mv testcov project_testcovs/testcov_$1  # Save per-repo test coverage bundle
mkdir testcov  # Empty dir for next bind mount
