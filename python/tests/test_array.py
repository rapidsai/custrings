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

print(".gather([1,3,5,7]):  ",strs.gather([1,3,5,7]))
arr = np.array([1,3,5,7],dtype=np.int32)
d_arr = rmm.to_device(arr)
devmem = d_arr.device_ctypes_pointer.value
print(".gather([1,3,5,7],4):",strs.gather(devmem,4))
print("[3]:",strs[3])
print("[[1,3,5,7]]:",strs[[1,3,5,7]])
print("[1:7:2]:",strs[1:7:2])
print("[7:1:-2]:",strs[7:1:-2])
print("[arr]:",strs[arr])
print("[d_arr]:",strs[d_arr])

print(".remove_strings([2,4,6,8]):  ",strs.remove_strings([2,4,6,8]))
d_arr = rmm.to_device(np.array([2,4,6,8],dtype=np.int32))
devmem = d_arr.device_ctypes_pointer.value
print(".remove_strings([2,4,6,8],4):",strs.remove_strings(devmem,4))

strs = None