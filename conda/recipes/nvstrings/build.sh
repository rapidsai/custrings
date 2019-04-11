#!/usr/bin/env bash

CMAKE_COMMON_VARIABLES=" -DCMAKE_INSTALL_PREFIX=$PREFIX -DCMAKE_BUILD_TYPE=Release"

# show environment
printenv
# Cleanup local git
git clean -xdf
# Change directory for build process
cd cpp
# Use CMake-based build procedure
mkdir build
cd build
# configure
cmake $CMAKE_COMMON_VARIABLES ..
# build/install cpp
make -j${PARALLEL_LEVEL} VERBOSE=1 install
# build/install python
cd ../../python
$PYTHON setup.py install --single-version-externally-managed --record=record.txt
