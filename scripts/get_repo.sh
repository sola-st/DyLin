#!/bin/sh

mkdir $3
cd $3
git init
git remote add origin $1
git fetch --depth 1 origin $2
git checkout $2
cd ..