#!/bin/bash
set -xe

conda build -c defaults -c conda-forge --python=${PYTHON} conda/recipes/nvstrings
