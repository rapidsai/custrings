NVStrings
=========

This repo is for CUDA string processing code for data science applications.
The modules here start with Pandas strings parity but also can include
more advanced features. 

This is a standalone library with no other dependencies but is expected to be used in RAPIDS applications.
And some RAPIDS modules may depend on NVStrings.

## Directories

### cpp

C/C++ source code including python bindings.
See [/cpp/README.md](cpp/README.md)

### python

Python modules and wrappers including test scripts.
See [/python/README.md](python/README.md)

### data

These are sample data files for testing the library features.
There is no source code here but this is considered an active directory since many of the test cases rely on them.

### docs

Documentation of python interfaces generated from the python source code.
This also includes some general documentation on capabilities and limitations.

### conda

Support files for deploying to conda environments.

## Development Setup

See the [C++ readme](cpp/README.md) for instructions building and installing.

### Docker

The Dockerfile can be used to setup a build and test environment.
It is based on the [cuDF](https://github.com/rapidsai/cudf) docker image mainly to keep a consistent conda enviornment.
The NVStrings library has no dependencies on cuDF.