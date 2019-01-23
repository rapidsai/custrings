#
import nvstrings
import numpy as np
from numba import cuda

#
s = nvstrings.to_device(["1234","5678","90",None,"-876","543.2","-0.12",".55","-.002","","de","abc123","123abc"])
print(s)
#
print(".stoi():",s.stoi())
arr = np.arange(s.size(),dtype=np.int32)
d_arr = cuda.to_device(arr)
s.stoi(d_arr.device_ctypes_pointer.value)
print(".stoi(devptr):",d_arr.copy_to_host())

#
print(".stof():",s.stof())
arr = np.arange(s.size(),dtype=np.float32)
d_arr = cuda.to_device(arr)
s.stof(d_arr.device_ctypes_pointer.value)
print(".stof(devptr):",d_arr.copy_to_host())

#
print(".hash():",s.hash())
arr = np.arange(s.size(),dtype=np.uint32)
d_arr = cuda.to_device(arr)
s.hash(d_arr.device_ctypes_pointer.value)
print(".hash(devptr):",d_arr.copy_to_host())

#
s = nvstrings.to_device(['1234567890', 'de', '1.75', '-34', '+9.8', '7¼', 'x³', '2³', '12⅝','','\t\r\n '])
print(s)
arr = np.arange(s.size(),dtype=np.byte)
d_arr = cuda.to_device(arr)

#
print(".isalnum():",s.isalnum())
s.isalnum(d_arr.device_ctypes_pointer.value)
print(".isalnum(devptr):",d_arr.copy_to_host())

#
print(".isalpha():",s.isalpha())
s.isalpha(d_arr.device_ctypes_pointer.value)
print(".isalpha(devptr):",d_arr.copy_to_host())

#
print(".isdigit():",s.isdigit())
s.isdigit(d_arr.device_ctypes_pointer.value)
print(".isdigit(devptr):",d_arr.copy_to_host())

#
print(".isdecimal():",s.isdecimal())
s.isdecimal(d_arr.device_ctypes_pointer.value)
print(".isdecimal(devptr):",d_arr.copy_to_host())

#
print(".isspace():",s.isspace())
s.isspace(d_arr.device_ctypes_pointer.value)
print(".isspace(devptr):",d_arr.copy_to_host())

#
print(".isnumeric():",s.isnumeric())
s.isnumeric(d_arr.device_ctypes_pointer.value)
print(".isnumeric(devptr):",d_arr.copy_to_host())
