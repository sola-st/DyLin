#!/bin/bash
python scripts/kaggle_evaluation.py --number 10 --competition $KAGGLE_COMPETITION --path /home/dylinuser/kaggle_files

# Copy report jsons
mkdir /home/dylinuser/results
cd /home/dylinuser/results
mkdir results_$KAGGLE_COMPETITION
cp /home/dylinuser/*.json results_$KAGGLE_COMPETITION

# Copy report csv
mkdir results_$KAGGLE_COMPETITION/table
cp /home/dylinuser/*.csv results_$KAGGLE_COMPETITION/table

# Copy downloaded submissions
mkdir results_$KAGGLE_COMPETITION/submissions
cp /home/dylinuser/kaggle_files/*.py results_$KAGGLE_COMPETITION/submissions
cp /home/dylinuser/kaggle_files/*.json results_$KAGGLE_COMPETITION/submissions