#!/usr/bin/env bash
cd python
$PYTHON setup.py install
cp ../cpp/include/NVStrings.h $PREFIX/include/nvstrings
cp ../cpp/include/NVCategory.h $PREFIX/include
cp build/lib.linux-x86_64*/libNVStrings.so $PREFIX/lib
cp build/lib.linux-x86_64*/libNVCategory.so $PREFIX/lib
