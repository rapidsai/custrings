#!/bin/bash
set -xe

conda build -c conda-forge -c defaults --python=${PYTHON} conda/recipes/nvstrings
