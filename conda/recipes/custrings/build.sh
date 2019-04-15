#!/usr/bin/env bash

CMAKE_COMMON_VARIABLES=" -DCMAKE_INSTALL_PREFIX=$PREFIX -DCMAKE_BUILD_TYPE=Release"

# show environment
printenv
# Cleanup local git
# git clean -xdf
# Change directory for build process
cd python
# Build Python extensions and install library
$PYTHON setup.py install --single-version-externally-managed --record=record.txt
