#!/bin/bash
git clone --depth 1 "$1" temp-linecount-repo &&
  printf "('temp-linecount-repo' will be deleted automatically)\n\n\n" &&
  pytest --collect-only temp-linecount-repo
rm -rf temp-linecount-repo