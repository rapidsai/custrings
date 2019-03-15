import os
import sys
import subprocess
import shutil

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir='../cpp'):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuildExt(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError('cmake is required')

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        output_dir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))

        abi_flag = os.environ.get('CMAKE_CXX11_ABI', 'OFF')
        build_type = 'Debug' if self.debug else 'Release'
        cmake_args = ['cmake',
                      ext.sourcedir,
                      '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + output_dir,
                      '-DCMAKE_BUILD_TYPE=' + build_type,
                      '-DCMAKE_CXX11_ABI=' + abi_flag]
        cmake_args.extend(
            [x for x in os.environ.get('CMAKE_COMMON_VARIABLES', '').split(' ')
             if x])

        env = os.environ.copy()
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(cmake_args,
                              cwd=self.build_temp,
                              env=env)
        subprocess.check_call(['make', '-j', ext.name],
                              cwd=self.build_temp,
                              env=env)
        print()


shutil.rmtree('build', ignore_errors=True)

install_requires = ['numba>=0.40.0dev']

with open('../LICENSE', encoding='UTF-8') as f:
    license_text = f.read()

cuda_version = ''.join(os.environ.get('CUDA_VERSION', 'unknown').split('.')[:2])
name = 'nvstrings-cuda{}'.format(cuda_version)
version = os.environ.get('GIT_DESCRIBE_TAG', '0.3.0.dev0').lstrip('v')
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

from pip_correction import convert_to_manylinux

convert_to_manylinux(name, version)
