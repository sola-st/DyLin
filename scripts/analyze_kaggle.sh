#!/bin/bash
python scripts/kaggle_evaluation.py --number 10 --competition $KAGGLE_COMPETITION --path /Work/kaggle_files --kaggleConf /Work/.kaggle

# Copy report jsons
mkdir /Work/results
cd /Work/results
mkdir results_$KAGGLE_COMPETITION
cp /Work/reports/*.json results_$KAGGLE_COMPETITION

# Copy report csv
mkdir results_$KAGGLE_COMPETITION/table
cp /Work/reports/*.csv results_$KAGGLE_COMPETITION/table

# Copy downloaded submissions
mkdir results_$KAGGLE_COMPETITION/submissions
cp /Work/kaggle_files/*.py results_$KAGGLE_COMPETITION/submissions
cp /Work/kaggle_files/*.py.orig results_$KAGGLE_COMPETITION/submissions
cp /Work/kaggle_files/*.json results_$KAGGLE_COMPETITION/submissions