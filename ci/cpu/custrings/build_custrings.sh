#!/bin/bash
set -xe

conda build --python=${PYTHON} conda/recipes/nvstrings
