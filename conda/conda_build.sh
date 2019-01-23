#!/usr/bin/env bash

# docker run -it --rm -v `pwd`:/CUDAStrings -w=/CUDAStrings --runtime=nvidia -e PYTHON=3.5 -e MY_UPLOAD_KEY=secret-key -e UPLOAD_PACKAGE=0 mikewendt/gpuci:miniconda3_cuda9.2-devel-ubuntu16.04_gcc5_gcxx5_py3.5 /CUDAStrings/conda_build.sh

set -xe

conda install conda-build anaconda-client conda-verify -y
conda build -c defaults -c conda-forge --python=${PYTHON} conda/recipes/nvstrings

if [ "$UPLOAD_PACKAGE" == '1' ]; then
    export UPLOADFILE=`conda build -c defaults -c conda-forge --python=${PYTHON} conda/recipes/nvstrings --output`
    SOURCE_BRANCH=master

    test -e ${UPLOADFILE}
    CUDA_REL=${CUDA:0:3}
    if [ "${CUDA:0:2}" == '10' ]; then
    # CUDA 10 release
    CUDA_REL=${CUDA:0:4}
    fi

    LABEL_OPTION="--label dev --label cuda${CUDA_REL}"
    if [ "${LABEL_MAIN}" == '1' ]; then
    LABEL_OPTION="--label main --label cuda${CUDA_REL}"
    fi
    echo "LABEL_OPTION=${LABEL_OPTION}"

    if [ -z "$MY_UPLOAD_KEY" ]; then
    echo "No upload key"
    return 0
    fi

    echo "Upload"
    echo ${UPLOADFILE}
    anaconda -t ${MY_UPLOAD_KEY} upload -u nvidia ${LABEL_OPTION} --force ${UPLOADFILE}
else
    echo "Skipping upload"
fi
