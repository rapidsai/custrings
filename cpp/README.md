NVStrings
===========

## API documentation

Current active code is following the API design: [https://confluence.nvidia.com/pages/viewpage.action?spaceKey=DL&title=NVStrings+class](https://confluence.nvidia.com/pages/viewpage.action?spaceKey=DL&title=NVStrings+class)

### Dependecies

The following has been used to build and test library.

* Linux 16.04
* CMake 3.12
* gcc 5.4
* CUDA 9.2, 10.0
* NVIDIA driver 396.44+  (Pascal arch or better)

### Building for Python 

Follow this section to use the C++ Python module.
This will build the string library dependencies automatically.
The above dependencies can be setup using conda and the `/dev_p37.yml` file:
```
    conda env create -n devstr conda/environments/dev_p37.yml
```
This .yml file is based on the ones from cudf and may have more packages then is actually needed for this module.

#### CMake for Python module

The CMake build requires the conda environment in order to find the dependent Python libraries. Build by creating a `build` dir under `/cpp` and calling cmake:
```
    source activate devstr
    cd cpp
    mkdir build
    cd build
    cmake ..
    make -j
```

Output includes `pyniNVStrings.so` and `libNVStrings.so`
Use it with `nvstrings.py` to run any of the `.py` files in the `python/tests` dir.
Output also includes `pyniNVCategory.so` and `libNVCategory.so`
Use them with `nvcategory.py` to run any of the `.py` files in the `python/tests` dir.

#### Makefile for Python module
The `Makefile.with_python` can be used to build standalone modules though they may be imcompatible with cuDF. Use the CMake as described above to ensure compatibility.

The `Makefile.with_python` requires environment variable `PYNI_PATH` to point to Python include and lib. For example:
```
    export PYNI_PATH=/home/user/anaconda3/envs/devstr
    
    # if you're using a RAPIDS container
    export PYNI_PATH=/conda/envs/cudf
```
The Python version must also be specified with the `PYTHON_VERSION` environment variable:
```
    export PYTHON_VERSION=3.7
```
```
    make -f Makefile.with_python
```
Output is `pyniNVStrings.so` and `libNVStrings.so` and `pyniNVCategory.so` and `libNVCategory.so`


#### Installing Python lib

The following will call cmake to build the modules and install/deploy them in the current environment:
```
    source activate devstr
    cd python
    python setup.py install
```


### Building for C++

Follow this section to build just the C++ classes directly without the Python modules.
```
    cd cpp
    make -j
```

Output is `libNVStrings.so` and `libNVCategory.so` only.
Use these along with `NVStrings.h` and `NVCategory.h`.
Example test file `/cpp/tests/csv.cu`


#### Makefile for NVStrings C++ interface

```
    cd cpp
    make -j
    sudo make install
```
The `make install` copies the `libNVStrings.so` and `libNVCategory.so` to `/usr/local/lib`
and `NVStrings.h` and `NVCategory.h` to `/usr/local/include/custr`.

Use `#include <custr/NVStrings.h>` in source files.
Use `-L/usr/local/lib -lNVStrings` for linking.

