#!/bin/bash
set -xe

# set env var for cmake to work
export CUDACXX=$CUDAHOSTCXX

conda build -c defaults -c conda-forge --python=${PYTHON} conda/recipes/nvstrings
