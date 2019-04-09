#!/bin/bash
# Copyright (c) 2019, NVIDIA CORPORATION.
#####################################
# cuStrings GPU build script for CI #
#####################################
set -ex

# Logger function for build status output
function logger() {
  echo -e "\n>>>> $@\n"
}

# Set path and build parallel level
export PATH=/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=4
export CUDA_REL=${CUDA_VERSION%.*}

# Define path to nvcc
export CUDACXX=/usr/local/cuda/bin/nvcc

# Enable ABI builds
export CMAKE_CXX11_ABI=ON

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

logger "Init conda..."
conda init
source ~/.bashrc

logger "Activate conda env..."
conda activate gdf
conda install -y librmm==0.7.*
pip install cmake_setuptools

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
# BUILD - from source
################################################################################

logger "Building custrings..."
mkdir -p $WORKSPACE/cpp/build
cd $WORKSPACE/cpp/build
logger "Run cmake custrings..."
cmake -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_CXX11_ABI=ON -DPython3_ROOT_DIR=$CONDA_PREFIX ..

logger "Clean up make..."
make clean

logger "Make custrings..."
make -j${PARALLEL_LEVEL}

logger "Install custrings cpp..."
make install

logger "Install custrings python..."
cd ../../python
pip install .

################################################################################
# TEST - something
################################################################################

logger "Check GPU usage..."
nvidia-smi

logger "Simple test..."
cd tests
python test_build.py
