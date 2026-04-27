for ((i=1; i<=37; i++)); do  # One static-linter pass per benchmark repo line
    bash run_single_linter.sh $i
done
