for ((i=36; i<=37; i++)); do  # Adjust loop bounds to run DyLin without analysis coverage on more indices
    bash run_single_project.sh $i nocov  # nocov = run analyses without DynaPyt coverage instrumentation
done
