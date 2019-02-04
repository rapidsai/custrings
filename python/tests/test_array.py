#
import nvstrings
import numpy as np

#
from librmm_cffi import librmm as rmm
from librmm_cffi import librmm_config as rmm_cfg
rmm_cfg.use_pool_allocator = True 
rmm.initialize()

#
strs = nvstrings.to_device(["abc","defghi",None,"jkl","mno","pqr","stu","dog and cat","accénted",""])
print(strs)

print(".sublist([1,3,5,7]):  ",strs.sublist([1,3,5,7]))
d_arr = rmm.to_device(np.array([1,3,5,7],dtype=np.int32))
devmem = d_arr.device_ctypes_pointer.value
print(".sublist([1,3,5,7],4):",strs.sublist(devmem,4))
print("[3]:",strs[3])
print("[[1,3,5,7]]:",strs[[1,3,5,7]])
print("[1:7:2]:",strs[1:7:2])

print(".remove_strings([2,4,6,8]):  ",strs.remove_strings([2,4,6,8]))
d_arr = rmm.to_device(np.array([2,4,6,8],dtype=np.int32))
devmem = d_arr.device_ctypes_pointer.value
print(".remove_strings([2,4,6,8],4):",strs.remove_strings(devmem,4))

strs = None