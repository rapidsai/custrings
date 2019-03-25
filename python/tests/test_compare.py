
import numpy as np
import nvstrings

#
from librmm_cffi import librmm as rmm
from librmm_cffi import librmm_config as rmm_cfg
rmm_cfg.use_pool_allocator = True 
rmm.initialize()

strs = nvstrings.to_device(["hello","there","world","accéntéd",None,""])
#
arr = np.arange(strs.size(),dtype=np.int32)
d_arr = rmm.to_device(arr)
devmem = d_arr.device_ctypes_pointer.value

print(strs)
print(".compare(there):",strs.compare("there"))
strs.compare("there",devmem)
print(".compare(there,devmem):",d_arr.copy_to_host())

print(".find(o):",strs.find("o"))
strs.find("o",devptr=devmem)
print(".find(o,devmem)",d_arr.copy_to_host())
print(".find_from(l,..)",strs.find_from("l",devmem))

print(".rfind(e):",strs.rfind("e"))
strs.rfind("e",devptr=devmem)
print(".rfind(e,devmem):",d_arr.copy_to_host())

print(".find(é):",strs.find("é"))
print(".rfind(é):",strs.rfind("é"))
print(".find():",strs.find(""))
print(".rfind():",strs.rfind(""))

print(".find_multiple(e,o,d):",strs.find_multiple(['e','o','d']))

print(".startswith(he):",strs.startswith("he"))
print(".endswith(d):",strs.endswith("d"))

strs2 = nvstrings.to_device(["hello","here",None,"accéntéd",None,""])
print(".match_strings(",strs2,"):",strs.match_strings(strs2))
strs2 = None

strs = nvstrings.to_device(["he-llo","-there-","world-","accént-éd",None,"-"])
#
arr = np.arange(strs.size(),dtype=np.int32)
d_arr = rmm.to_device(arr)
devmem = d_arr.device_ctypes_pointer.value

print(strs)
print(".index(-):",strs.index("-"))
strs.index("-",devptr=devmem)
print(".index(-,devmem):",d_arr.copy_to_host())

print(".rindex(-):",strs.rindex("-"))
strs.rindex("-",devptr=devmem)
print(".rindex(-,devmem):",d_arr.copy_to_host())

#
arr = np.arange(strs.size(),dtype=np.byte)
d_arr = rmm.to_device(arr)
devmem = d_arr.device_ctypes_pointer.value
print(".contains(l):",strs.contains("l"))
strs.contains("l",devptr=devmem)
print(".contains(l,devmem):",d_arr.copy_to_host())

strs = None