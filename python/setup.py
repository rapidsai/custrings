import os
import sys
import subprocess
import shutil

from setuptools import setup

from cmake_setuptools import CMakeExtension, CMakeBuildExt, convert_to_manylinux

shutil.rmtree('build', ignore_errors=True)

install_requires = ['numba>=0.40.0dev']

with open('../LICENSE', encoding='UTF-8') as f:
    license_text = f.read()

cuda_version = ''.join(os.environ.get('CUDA_VERSION', 'unknown').split('.')[:2])
name = 'nvstrings-cuda{}'.format(cuda_version)
version = os.environ.get('GIT_DESCRIBE_TAG', '0.0.0.dev0').lstrip('v')
setup(name=name,
      description='CUDA strings Python bindings',
      version=version,
      py_modules=['nvstrings', 'nvcategory'],
      url='https://github.com/NVIDIA/nvstrings',
      author='NVIDIA Corporation',
      license=license_text,
      install_requires=install_requires,
      ext_modules=[CMakeExtension('NVStrings'),
                   CMakeExtension('pyniNVStrings'),
                   CMakeExtension('NVCategory'),
                   CMakeExtension('pyniNVCategory')],
      cmdclass={'build_ext': CMakeBuildExt},
      headers=['../cpp/include/NVStrings.h', '../cpp/include/NVCategory.h'],
      zip_safe=False
      )

convert_to_manylinux(name, version)
