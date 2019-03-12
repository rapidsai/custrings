#!/bin/bash
# Copyright (c) 2019, NVIDIA CORPORATION.
###########################################
# cuStrings CPU conda build script for CI #
###########################################
set -ex

# Logger function for build status output
function logger() {
  echo -e "\n>>>> $@\n"
}

# Set path and build parallel level
export PATH=/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=4

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

logger "Check versions..."
python --version
$CC --version
$CXX --version
conda list

# FIX Added to deal with Anancoda SSL verification issues during conda builds
conda config --set ssl_verify False

################################################################################
# INSTALL - Install NVIDIA driver
################################################################################

# Nothing to install yet

################################################################################
# BUILD - Conda package builds (conda deps: libcuml <- cuml)
################################################################################

logger "Build conda pkg for custrings..."
source ci/cpu/custrings/build-custrings.sh

################################################################################
# UPLOAD - Conda packages
################################################################################

logger "Upload conda pkgs for custrings..."
source ci/cpu/custrings/upload-anaconda.sh
