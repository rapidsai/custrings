#!/bin/bash

#Copyright (c) 2019, NVIDIA CORPORATION.

# custrings build script

# This script is used to build the component(s) in this repo from
# source, and can be called with various options to customize the
# build as needed (see the help output for details)

# Abort script on first error
set -e

NUMARGS=$#
ARGS=$*

 # Get root of git repository without assuming location of build.sh script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$SCRIPT_DIR"
REPODIR=$(git rev-parse --show-toplevel)

VALIDARGS="clean libcustrings custrings -v -g -n -h"
HELP="$0 [clean] [libcustrings] [custrings] [-v] [-g] [-n] [-h]
   clean        - remove all existing build artifacts and configuration (start over)
   libcustrings - build the custrings C++ code only
   custrings    - build the custrings Python package
   -v           - verbose build mode
   -g           - build for debug
   -n           - no install step
   -h           - print this text
    default action (no args) is to build and install 'libcustrings' then 'custrings' targets
"
LIBCUSTRINGS_BUILD_DIR=${REPODIR}/cpp/build
CUSTRINGS_BUILD_DIR=${REPODIR}/python
BUILD_DIRS="${LIBCUSTRINGS_BUILD_DIR}"

 # Set defaults for vars modified by flags to this script
VERBOSE=""
BUILD_TYPE=Release
INSTALL_TARGET=install

 # Set defaults for vars that may not have been defined externally
#  FIXME: if INSTALL_PREFIX is not set, check PREFIX, then check
#         CONDA_PREFIX, but there is no fallback from there!
INSTALL_PREFIX=${INSTALL_PREFIX:=${PREFIX:=${CONDA_PREFIX}}}
PARALLEL_LEVEL=${PARALLEL_LEVEL:=""}
PYTHON=${PYTHON:-python}

 function hasArg {
    (( NUMARGS != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

function createBuildArea {
    mkdir -p "${LIBCUSTRINGS_BUILD_DIR}"
    cd "${LIBCUSTRINGS_BUILD_DIR}"
    cmake -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
          -DCMAKE_CXX11_ABI=ON \
          -DCMAKE_BUILD_TYPE=${BUILD_TYPE} ..
}

 if hasArg -h; then
    echo "${HELP}"
    exit 0
fi

 # Check for valid usage
if (( NUMARGS != 0 )); then
    for a in ${ARGS}; do
        if ! (echo " ${VALIDARGS} " | grep -q " ${a} "); then
            echo "Invalid option: ${a}"
            exit 1
        fi
    done
fi

 # Process flags
if hasArg -v; then
    VERBOSE=1
    set -x
fi
if hasArg -g; then
    BUILD_TYPE=Debug
fi
if hasArg -n; then
    INSTALL_TARGET=""
fi

 # If no args or clean given, run it prior to any other steps
 if (( NUMARGS == 0 )) || hasArg clean; then
    # If the dirs to clean are mounted dirs in a container, the
    # contents should be removed but the mounted dirs will remain.
    # The find removes all contents but leaves the dirs, the rmdir
    # attempts to remove the dirs but can fail safely.
    for bd in ${BUILD_DIRS}; do
        if [ -d "${bd}" ]; then
            find "${bd}" -mindepth 1 -delete
            rmdir "${bd}" || true
        fi
    done
fi

################################################################################
# Configure, build, and install libcustrings
if hasArg libcustrings; then
    createBuildArea
    make -j${PARALLEL_LEVEL} VERBOSE=${VERBOSE}
    if [[ ${INSTALL_TARGET} != "" ]]; then
        make -j${PARALLEL_LEVEL} VERBOSE=${VERBOSE} install
    fi
fi

 # Build and install the custrings Python package
if (( NUMARGS == 0 )) || hasArg custrings; then
    # Build and install libcustrings.so 
    createBuildArea
    make -j${PARALLEL_LEVEL} VERBOSE=${VERBOSE}
    make -j${PARALLEL_LEVEL} VERBOSE=${VERBOSE} install

    # build custrings
    cd "$CUSTRINGS_BUILD_DIR"
    $PYTHON setup.py install --single-version-externally-managed --record=record.txt
fi
