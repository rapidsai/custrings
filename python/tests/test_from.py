
import numpy as np
from numba import cuda
import nvstrings

strs = nvstrings.to_device(["hello world","holy accéntéd","batman",None,""])
#
arr = np.arange(strs.size(),dtype=np.int32)
d_arr = cuda.to_device(arr)
devmem = d_arr.device_ctypes_pointer.value

print(strs)

print(".find(o):",strs.find("o"))
strs.find("o",devptr=devmem)
print(".find(o,devmem)",d_arr.copy_to_host())
print(".find_from(l,..)",strs.find_from("l",devmem))

print(".slice_from(..)",strs.slice_from(devmem))