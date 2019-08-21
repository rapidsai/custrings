#!/bin/bash
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
export RMM_VERSION=0.7.*

# Define path to nvcc
export CUDACXX=/usr/local/cuda/bin/nvcc

# Set home to the job's workspace
export HOME=$WORKSPACE

################################################################################
# SETUP - Check environment
################################################################################

logger "Get env..."
env

logger "Activate conda env..."
source activate gdf
conda install -c rapidsai/label/cuda$CUDA_REL -c rapidsai-nightly/label/cuda$CUDA_REL \
    rmm=$RMM_VERSION cmake_setuptools\>=0.1.3
    
pip install sphinx sphinx_rtd_theme sphinxcontrib-websupport sphinx-markdown-tables \
    numpydoc ipython recommonmark

logger "Check versions..."
python --version
$CC --version
$CXX --version
$CUDACXX --version
conda config --get channels
conda list
conda config --set ssl_verify False

logger "Check GPU usage..."
nvidia-smi

################################################################################
# BUILD - from source
################################################################################

logger "Building custrings..."
mkdir -p $WORKSPACE/cpp/build
cd $WORKSPACE/cpp/build
logger "Run cmake custrings..."
cmake -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_CXX11_ABI=ON ..

logger "Clean up make..."
make clean

logger "Make custrings..."
make -j${PARALLEL_LEVEL}

logger "Install custrings cpp..."
make -j${PARALLEL_LEVEL} install

logger "Install custrings python..."
cd ../../python
python setup.py install --single-version-externally-managed --record=record.txt

logger "Build cuStrings docs..."
cd $WORKSPACE/docs
make html

rm -rf /data/docs/html/*
mv build/html/* /data/docs/html
