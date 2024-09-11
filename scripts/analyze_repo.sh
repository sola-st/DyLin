#!/bin/bash

mkdir /tmp/dynapyt_output-1234-abcd
if [ $2 == "cov" ]; then
    mkdir /tmp/dynapyt_coverage-1234-abcd
    mkdir /Work/reports/dynapyt_coverage
fi
python DyLin/scripts/prepare_repo.py --repo $1 --config /Work/DyLin/dylin_config_project.txt
if [ $2 == "cov" ]; then
    python DyLin/scripts/analyze_repo.py --repo $1 --config /Work/DyLin/dylin_config_project.txt
else
    python DyLin/scripts/analyze_repo.py --repo $1 --config /Work/DyLin/dylin_config_project.txt --no-cov
fi
mkdir /Work/reports/dynapyt_output

cp -r /tmp/dynapyt_output-1234-abcd /Work/reports/dynapyt_output
if [ $2 == "cov" ]; then
    cp -r /tmp/dynapyt_coverage-1234-abcd /Work/reports/dynapyt_coverage
fi
