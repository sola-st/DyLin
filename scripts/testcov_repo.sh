#!/bin/bash

python DyLin/scripts/prepare_testcov.py --repo $1
timeout 1h python DyLin/scripts/testcov_repo.py --repo $1