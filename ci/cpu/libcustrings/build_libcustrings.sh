#!/bin/bash
set -e

echo "Buildings libcustrings"
CUDA_REL=${CUDA:0:3}
if [ "${CUDA:0:2}" == '10' ]; then
    # CUDA 10 release
    CUDA_REL=${CUDA:0:4}
fi

conda build --python=${PYTHON} conda/recipes/libcustrings
