#!/usr/bin/env bash

# Copyright (c) 2018-2019, NVIDIA CORPORATION.

# show environment
printenv
# Cleanup local git
git clean -xdf
# This assumes the script is executed from the root of the repo directory
./build.sh -v libcustrings
