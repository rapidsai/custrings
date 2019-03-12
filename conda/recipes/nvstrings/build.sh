#!/usr/bin/env bash
cd python
$PYTHON setup.py install
cp NVStrings.h $PREFIX/include
cp NVCategory.h $PREFIX/include
cp build/lib.linux-x86_64*/libNVStrings.so $PREFIX/lib
cp build/lib.linux-x86_64*/libNVCategory.so $PREFIX/lib
