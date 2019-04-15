#!/usr/bin/env bash

#Build custrings once per PYTHON
if [[ "$CUDA" == "9.2" ]]; then
    export BUILD_CUSTRINGS=1
else
    export BUILD_CUSTRINGS=0
fi

#Build libcustrings once per CUDA
if [[ "$PYTHON" == "3.6" ]]; then
    export BUILD_LIBCUSTRINGS=1
else
    export BUILD_LIBCUSTRINGS=0
fi
