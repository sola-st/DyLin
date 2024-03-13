#!/bin/bash

python DyLin/scripts/prepare_repo.py --repo $1
timeout 1h python DyLin/scripts/analyze_repo.py --repo $1
