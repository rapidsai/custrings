# <div align="left"><img src="img/rapids_logo.png" width="90px"/>&nbsp;cuStrings - GPU String Manipulation</div>

[![Build Status](http://18.191.94.64/buildStatus/icon?job=custrings-master)](http://18.191.94.64/job/custrings-master/)&nbsp;&nbsp;[![Documentation Status](https://readthedocs.org/projects/nvstrings/badge/?version=latest)](https://rapidsai.github.io/projects/custrings/en/latest)

**NOTE:** For the latest stable [README.md](https://github.com/rapidsai/custrings/blob/master/README.md) ensure you are on the `master` branch.

Built with Pandas DataFrame's columnar string operations in mind, cuStrings is a GPU string manipulation library for splitting, applying regexes, concatenating, replacing tokens, etc in arrays of strings.

nvStrings (the Python bindings for cuStrings), provides a pandas-like API that will be familiar to data engineers & data scientists, so they can use it to easily accelerate their workflows without going into the details of CUDA programming.

For example, the following snippet loads a CSV, then uses the GPU to perform replacements typical in data-preparation tasks.
```python
import nvstrings, nvcategory
import requests

url="https://github.com/plotly/datasets/raw/master/tips.csv"
content = requests.get(url).content.decode('utf-8')

#split content into a list, remove header
host_lines = content.strip().split('\n')[1:]

#copy strings to gpu
gpu_lines = nvstrings.to_device(host_lines)

#split into columns on gpu
gpu_columns = gpu_lines.split(',')
gpu_day_of_week = gpu_columns[4]

#use gpu `replace` to re-encode tokens on GPU
for idx, day in enumerate(['Sun', 'Mon', 'Tues', 'Wed', 'Thur', 'Fri', 'Sat']):
    gpu_day_of_week = gpu_day_of_week.replace(day, str(idx))

# or, use nvcategory's builtin GPU categorization
cat = nvcategory.from_strings(columns[4])

# copy category keys to host and print
print(cat.keys())

# copy "cleaned" strings to host and print
print(gpu_day_of_week)
```

Output:
```
['Fri', 'Sat', 'Sun', 'Thur']

# many entries omitted for brevity
[2, 2, 2, 2, ..., 1, 1, 1, 3]
```

cuStrings is a standalone library with no other dependencies. Other RAPIDS projects (like cuDF) depend on cuStrings and its nvStrings Python bindings.

For more examples, see [Python API documentation](http://rapidsai.github.io/projects/nvstrings/en/latest), and [cuStrings CUDA/C++ API](cpp/cuStrings-API.pdf).
## Quick Start

Please see the [Demo Docker Repository](https://hub.docker.com/r/rapidsai/rapidsai/), choosing a tag based on the NVIDIA CUDA version you’re running. This provides a ready to run Docker container with example notebooks and data, showcasing how you can utilize cuStrings.

## Installation

### Conda

cuStrings can be installed with conda ([miniconda](https://conda.io/miniconda.html), or the full [Anaconda distribution](https://www.anaconda.com/download)) from the `rapidsai` channel:
```bash
# for CUDA 9.2
conda install -c nvidia -c rapidsai -c numba -c conda-forge -c defaults \
    nvstrings=0.3 python=3.6

# or, for CUDA 10.0
conda install -c nvidia/label/cuda10.0 -c rapidsai/label/cuda10.0 -c numba \
    -c conda-forge -c defaults nvstrings=0.3 python=3.6
```

We also provide [nightly conda packages](https://anaconda.org/rapidsai-nightly) built from the tip of our latest development branch.

### Pip

nvstrings can also be installed from [PyPi](https://pypi.org/project/nvstrings).

```bash
# for CUDA 9.2
python3.6 -m pip install nvstrings-cuda92==0.3

# or, for CUDA 10.0
python3.6 -m pip install nvstrings-cuda100==0.3
```

Note: cuStrings is supported only on Linux, and with Python versions 3.6 or 3.7.

See the [Get RAPIDS version picker](https://rapids.ai/start.html) for more OS and version info. 

## Build/Install from Source
See build [instructions](CONTRIBUTING.md#setting-up-your-build-environment).

## Contributing

Please see our [guide for contributing to cuStrings](CONTRIBUTING.md).

## Contact

Find out more details on the [RAPIDS site](https://rapids.ai/community.html)

## <div align="left"><img src="img/rapids_logo.png" width="265px"/></div> Open GPU Data Science

The RAPIDS suite of open source software libraries aim to enable execution of end-to-end data science and analytics pipelines entirely on GPUs. It relies on NVIDIA® CUDA® primitives for low-level compute optimization, but exposing that GPU parallelism and high-bandwidth memory speed through user-friendly Python interfaces.
