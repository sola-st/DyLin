#!/bin/bash

python DyLin/scripts/prepare_testcov.py --repo $1
python DyLin/scripts/testcov_repo.py --repo $1