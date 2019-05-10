#!/usr/bin/env bash

#Upload custrings once per PYTHON
if [[ "$CUDA" == "9.2" ]]; then
    export UPLOAD_CUSTRINGS=1
else
    export UPLOAD_CUSTRINGS=0
fi

#Upload libcustrings once per CUDA
if [[ "$PYTHON" == "3.6" ]]; then
    export UPLOAD_LIBCUSTRINGS=1
else
    export UPLOAD_LIBCUSTRINGS=0
fi
