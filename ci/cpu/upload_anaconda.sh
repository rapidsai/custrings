#!/bin/bash
# Copyright (c) 2019, NVIDIA CORPORATION.

set -e

export LIBCUSTRINGS_FILE=`conda build conda/recipes/libcustrings --output`
export CUSTRINGS_FILE=`conda build --python=$PYTHON conda/recipes/custrings --output`

SOURCE_BRANCH=master
CUDA_REL=${CUDA_VERSION%.*}

# Restrict uploads to master branch
if [ ${GIT_BRANCH} != ${SOURCE_BRANCH} ]; then
    echo "Skipping upload"
    return 0
fi

if [ -z "$MY_UPLOAD_KEY" ]; then
    echo "No upload key"
    return 0
fi

if [ "$UPLOAD_LIBCUSTRINGS" == '1' ]; then

  LABEL_OPTION="--label main --label cuda${CUDA_REL}"
  echo "LABEL_OPTION=${LABEL_OPTION}"

  echo "Upload"
  echo ${LIBCUSTRINGS_FILE}
  anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-rapidsai} ${LABEL_OPTION} --force ${LIBCUSTRINGS_FILE}
fi

if [ "$UPLOAD_CUSTRINGS" == '1' ]; then
    
    # Have to label all CUDA versions due to the compatibility to work with any CUDA
    LABEL_OPTION="--label main --label cuda9.2 --label cuda10.0"
    echo "LABEL_OPTION=${LABEL_OPTION}"

    echo "Upload"
    echo ${CUSTRINGS_FILE}
    anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-rapidsai} ${LABEL_OPTION} --force ${CUSTRINGS_FILE}
fi
