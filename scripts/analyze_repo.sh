#!/bin/bash

python DyLin/scripts/prepare_repo.py --repo $1
python DyLin/scripts/analyze_repo.py --repo $1
