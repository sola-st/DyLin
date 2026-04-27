for ((i=36; i<=37; i++)); do  # Adjust bounds to collect test coverage for more projects
    bash run_single_testcov.sh $i
done
