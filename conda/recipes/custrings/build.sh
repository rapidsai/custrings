#!/usr/bin/env bash

# show environment
printenv
# Cleanup local git
git clean -xdf
# This assumes the script is executed from the root of the repo directory
./build.sh custrings