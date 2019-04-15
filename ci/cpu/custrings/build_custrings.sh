#!/bin/bash
set -e

if [ "$BUILD_CUSTRINGS" == "1" ]; then
    echo "Building custrings"
    export CUSTRINGS_BUILD_NO_GPU_TEST=1

    conda build --python=${PYTHON} conda/recipes/custrings
fi
