
import numpy as np
import nvstrings, rave

#
from librmm_cffi import librmm as rmm
from librmm_cffi import librmm_config as rmm_cfg
rmm_cfg.use_pool_allocator = True 
rmm.initialize()

strs = nvstrings.to_device(["the quick brown fox jumped over the lazy brown dog","the sable siamésé cat jumped under the brown sofa",None,""])
#

print(strs)

print("token_count:",rave.token_count(strs))
d_arr = rmm.to_device(np.arange(strs.size(),dtype=np.int32))
rave.token_count(strs,' ',devptr=d_arr.device_ctypes_pointer.value)
print(" ",d_arr.copy_to_host())

tokens = rave.unique_tokens(strs)
print("unique_tokens:",tokens)

print("contains_strings:",rave.contains_strings(strs,tokens))
d_arr = rmm.to_device(np.arange(strs.size()*tokens.size(),dtype=np.byte))
rave.contains_strings(strs,tokens,devptr=d_arr.device_ctypes_pointer.value)
print(" ",d_arr.copy_to_host())

print("strings_counts:",rave.strings_counts(strs,tokens))
d_arr = rmm.to_device(np.arange(strs.size()*tokens.size(),dtype=np.int32))
rave.strings_counts(strs,tokens,devptr=d_arr.device_ctypes_pointer.value)
print(" ",d_arr.copy_to_host())

#
print("contains_strings(cat,dog,bird,the):",rave.contains_strings(strs,['cat','dog','bird','the']))
print("strings_counts(cat,dog,bird,the):",rave.strings_counts(strs,['cat','dog','bird','the']))

strs = None
tokens = None