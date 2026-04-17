for ((i=37; i<=38; i++)); do  # Adjust loop bounds for analysis-with-coverage runs
    bash run_single_project.sh $i cov  # cov = collect analysis coverage under dynapyt_coverage/
done
