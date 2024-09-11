#!/bin/bash
python scripts/kaggle_prepare.py --number 5 --competition $KAGGLE_COMPETITION --path /Work/kaggle_files --kaggleConf /Work/.kaggle

cd /Work/kaggle_files
for f in *.py; do
    sessionID=$(python -c "from uuid import uuid4; print(str(uuid4()))")
    cp /Work/dylin_config_kaggle.txt /tmp/dynapyt_analyses-$sessionID.txt
    sed -i "s|\$|;output_dir=/tmp/dynapyt_output-${sessionID}|" /tmp/dynapyt_analyses-$sessionID.txt
    (DYNAPYT_SESSION_ID=$sessionID timeout 10m python $f && echo "$f completed.") || echo "$f timed out!" &
done

# Wait for all background processes to finish
wait

for f in /tmp/dynapyt_output-*; do
    ls $f
    python -m dynapyt.post_run --output_dir=$f
done

# Copy report jsons
cd /Work/results
mkdir results_$KAGGLE_COMPETITION

# Copy report csv
mkdir results_$KAGGLE_COMPETITION/table
cp -r /tmp/dynapyt_output-* results_$KAGGLE_COMPETITION/table

mkdir results_$KAGGLE_COMPETITION/coverage
cp -r /tmp/dynapyt_coverage-* results_$KAGGLE_COMPETITION/coverage

# cp /tmp/dynapyt_output-*/*.json results_$KAGGLE_COMPETITION/table

# Copy downloaded submissions
mkdir results_$KAGGLE_COMPETITION/submissions
cp /Work/kaggle_files/*.py results_$KAGGLE_COMPETITION/submissions
cp /Work/kaggle_files/*.py.orig results_$KAGGLE_COMPETITION/submissions
cp /Work/kaggle_files/*.json results_$KAGGLE_COMPETITION/submissions