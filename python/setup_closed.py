from setuptools import setup

setup (name = 'nvstrings',
   description = 'CUDA strings Python bindings',
   version = '0.0.1',
   py_modules=['nvstrings', 'nvcategory', 'rave'],
   data_files=[('', ['build/pyniNVStrings.so', 'build/pyniNVCategory.so', 'build/pyniRave.so'])],
   zip_safe=False
)
