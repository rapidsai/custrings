cuStrings
===========

### API Design

[cuStrings API](cuStrings-API.pdf)

### Dependecies

The following has been used to build and test library.

* Linux 16.04
* CMake 3.12
* gcc 5.4
* CUDA 9.2, 10.0
* NVIDIA driver 396.44+  (Pascal arch or better)

This library is also dependent on the rapids/RMM library available at https://github.com/rapidsai/rmm
You must build and install RMM first before you can build this library.
Once built, set the following environment variable to where RMM is installed:
```
    export RMM_ROOT=/usr/local/include/rmm
```
This may be different if you installed this into a conda environment.
For example:
```
    export RMM_ROOT=${CONDA_PREFIX}/lib
```
Where the `CONDA_PREFIX` is the root directory of your target conda environment.

### Building for Python - Conda Environment

Follow this section to use the C++ Python module.
This will build the string library dependencies automatically.
The above dependencies can be setup using conda and with the `custring_dev.yml` file:
```
    conda env create -n devstr conda/environments/custrings_dev.yml
```
This will create a conda environment called `devstr` that can be used when building and running nvstrings.

### CMake for C++ interface

Build by creating a `build` dir under `/cpp` and calling cmake:
```
    cd cpp
    mkdir build
    cd build
    cmake ..
    make -j
```

Output includes `libNVStrings.so` and `libNVCategory.so` in the build dir.
You can install these into your `/usr/local` by calling
```
    sudo make install
```

If you want to build this for your conda environment only then run the cmake as follows instead:
```
    source activate ${CONDA_ENV}
    cmake .. -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}
    make -j install
```
Where the `CONDA_ENV` is `devstr` as described above or whatever conda environment name you are using.
And the `CONDA_PREFIX` is the conda root directory like `~/conda/envs/devstr` or what is appropriate for your conda environment.

#### C++ API Documentation

The C++ interface is documented using doxygen. After running the cmake as described above, you can
build the doxygen files using the following:

```
    make doc
```

This will create an `cpp/doxygen/html` directory with the API documentation.
You can load the `cpp/doxygen/html/index.html` into your browser to navigate through the documentation.


### Building/Installing Python Modules

If you installed `libNVStrings.so` in your `/usr/local/lib` path then you may need to set the following
environment variable:
```
    NVSTRINGS_ROOT=/usr/local/include/nvstrings
```

The following will build the modules and install/deploy them in the current environment:
```
    source activate ${CONDA_ENV}
    cd python
    python setup.py install
```

Output includes `pyniNVStrings.so` and `pyniNVCategory.so` in the build dir
which will also be deployed correctly into your conda environment.


### Docker

There is also a Dockerfile that can be used to setup a build and test environment.
It is based on the [cuDF](https://github.com/rapidsai/cudf) docker image mainly to keep a consistent conda environment.
To build this docker image you must first build or pull the appropriate cudf docker image.

Build the nvstrings docker image using:
```
    docker build -t custrings:latest .
```
To run the container use the following:
```
    docker run --runtime=nvidia -it custrings:latest bash
```
The image includes a Jupyter server. To run this server start the container as follows:
```
    docker run --runtime=nvidia -it -p 8889:8888 custrings:latest
```
Where 8889 is an open port on your local system. Leave off the command (e.g. bash) to start the server.
The Jupyter server is setup with key name 'rapids'.
