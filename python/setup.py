
import os

from setuptools import setup, find_packages
from pip_correction import convert_to_manylinux

install_requires = []

cuda_version = ''.join(os.environ.get('CUDA_VERSION', 'unknown')
                         .split('.')[:2])
name = 'nvstrings-cuda{}'.format(cuda_version)
version = os.environ.get('GIT_DESCRIBE_TAG', '0.0.0.dev0').lstrip('v')
setup(name=name,
      description='CUDA strings Python bindings',
      version=version,
      py_modules=['nvstrings', 'nvcategory', 'nvtext'],
      url='https://github.com/rapidsai/custrings',
      author='NVIDIA Corporation',
      license="Apache",
      install_requires=install_requires,
      packages=find_packages(),
      package_data={
          '': ['pyniNVStrings.so', 'pyniNVCategory.so', 'pyniNVText.so'],
      },
      include_package_data=True,
      zip_safe=False
      )

convert_to_manylinux(name, version)
