# https://github.com/rapidsai/cudf/tree/master/Dockerfile
FROM cudf

RUN apt-get update && apt-get install -y vim locales

# Add source to image
ADD cpp /nvstrings/cpp
ADD python /nvstrings/python
ADD LICENSE /nvstrings/LICENSE
ADD thirdparty /nvstrings/thirdparty
WORKDIR /nvstrings/cpp

ENV CONDA_ENV=cudf

RUN locale-gen en_US.UTF-8  
ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8

# Remove the cudf installed nvstrings
RUN source activate ${CONDA_ENV} && \
    conda remove nvstrings && \
    conda remove libnvstrings && \
    pip install cmake_setuptools

# Build
RUN source activate ${CONDA_ENV} && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} && make -j install

# Install
WORKDIR /nvstrings/python
RUN source activate ${CONDA_ENV} && python setup.py install

# Setup jupyter
WORKDIR /nvstrings/python/notebooks
RUN source activate ${CONDA_ENV} && conda install jupyterlab
CMD source activate ${CONDA_ENV} && jupyter-lab --allow-root --ip='0.0.0.0' --NotebookApp.token='rapids'

