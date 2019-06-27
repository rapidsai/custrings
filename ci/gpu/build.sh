#!/bin/bash
# Copyright (c) 2019, NVIDIA CORPORATION.
#####################################
# cuStrings GPU build script for CI #
#####################################
set -ex
NUMARGS=$#
ARGS=$*

# Logger function for build status output
function logger() {
  echo -e "\n>>>> $@\n"
}

# Arg parsing function
function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

# Set path and build parallel level
export PATH=/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=4
export CUDA_REL=${CUDA_VERSION%.*}

# Define path to nvcc
export CUDACXX=/usr/local/cuda/bin/nvcc

# Set home to the job's workspace
export HOME=$WORKSPACE

# Switch to project root; also root of repo checkout
cd $WORKSPACE

# Get latest tag and number of commits since tag
export GIT_DESCRIBE_TAG=`git describe --abbrev=0 --tags`
export GIT_DESCRIBE_NUMBER=`git rev-list ${GIT_DESCRIBE_TAG}..HEAD --count`

################################################################################
# SETUP - Check environment
################################################################################

logger "Get env..."
env

logger "Activate conda env..."
source activate gdf
conda install -y 'rmm==0.7.*' 'cmake_setuptools>=0.1.3'

logger "Check versions..."
python --version
$CC --version
$CXX --version
$CUDACXX --version
conda config --get channels
conda list

# FIX Added to deal with Anancoda SSL verification issues during conda builds
conda config --set ssl_verify False

################################################################################
# INSTALL - Install NVIDIA driver
################################################################################

logger "Check GPU usage..."
nvidia-smi

#logger "Install NVIDIA driver for CUDA $CUDA..."
#apt-get update -q
#DRIVER_VER="396.44-1"
#LIBCUDA_VER="396"
#if [ "$CUDA" == "10.0" ]; then
#  DRIVER_VER="410.72-1"
#  LIBCUDA_VER="410"
#fi
#DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
#  cuda-drivers=${DRIVER_VER} libcuda1-${LIBCUDA_VER}

################################################################################
# BUILD - Build libcuStrings and cuStrings
################################################################################

logger "Building libcustrings and custrings..."
${WORKSPACE}/build.sh 

################################################################################
# TEST - Test custrings
################################################################################

if hasArg --skip-tests; then
    logger "Skipping Tests..."
    exit 0
fi

logger "Check GPU usage..."
nvidia-smi

logger "Simple test..."
cd tests
pytest --cache-clear --junitxml=${WORKSPACE}/junit-custrings.xml -v
