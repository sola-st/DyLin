for ((i=2; i<=30; i++)); do  # Baseline sweep over project indices
    bash run_single_baseline.sh $i
done
