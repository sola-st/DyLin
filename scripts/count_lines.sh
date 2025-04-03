#!/bin/bash

INPUT_FILE="$1"
TEMP_DIR="$(mktemp -d)"

if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: Input file not found!"
    exit 1
fi

while read -r repo_url commit_hash _; do
    WORK_DIR="$TEMP_DIR/repo"
    rm -rf "$WORK_DIR"
    mkdir -p "$WORK_DIR"
    
    echo "Processing $repo_url at commit $commit_hash"
    
    git clone --quiet "$repo_url" "$WORK_DIR" || { echo "Failed to clone $repo_url"; continue; }
    cd "$WORK_DIR" || continue
    git checkout --quiet "$commit_hash" || { echo "Failed to checkout $commit_hash"; continue; }
    
    LINE_COUNT=$(sloccount . 2>/dev/null | grep -oP "python:\s+\K\d+" || echo "0")
    echo "Lines of code: $LINE_COUNT"
    # TEST_COUNT=$(pytest --collect-only 2>/dev/null | grep -oP "collected \K\d+" || echo "0")
    # echo "Tests found: $TEST_COUNT"
    
    cd - > /dev/null
    rm -rf "$WORK_DIR"
done < "$INPUT_FILE"

rm -rf "$TEMP_DIR"