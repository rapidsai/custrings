
import os

from setuptools import setup, find_packages
from pip_correction import convert_to_manylinux

install_requires = []

cuda_version = ''.join(os.environ.get('CUDA_VERSION', 'unknown')
                         .split('.')[:2])
name = 'nvstrings-cuda{}'.format(cuda_version)
version = os.environ.get('GIT_DESCRIBE_TAG', '0.0.0.dev0').lstrip('v')
setup(
    name=name,
    description='CUDA strings Python bindings',
    url='https://github.com/rapidsai/custrings',
    version=version,
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7"
    ],
    py_modules=['nvstrings', 'nvcategory', 'nvtext'],
    author='NVIDIA Corporation',
    license="Apache",
    install_requires=install_requires,
    data_files=[('', ['pyniNVStrings.so', 'pyniNVCategory.so', 'pyniNVText.so'])],
    packages=[''],
    package_data={
        '': [
            'pyniNVStrings.so', 'pyniNVCategory.so', 'pyniNVText.so'
        ],
    },
    include_package_data=True,
    zip_safe=False
)

convert_to_manylinux(name, version)
