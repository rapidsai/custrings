
import numpy as np
import nvstrings, nvtext

#
from librmm_cffi import librmm as rmm
from librmm_cffi import librmm_config as rmm_cfg
rmm_cfg.use_pool_allocator = True 
rmm.initialize()

strs = nvstrings.to_device(["the quick brown fox jumped over the lazy brown dog","the sable siamésé cat jumped under the brown sofa",None,""])
#

print(strs)

print("token_count:",nvtext.token_count(strs))
d_arr = rmm.to_device(np.arange(strs.size(),dtype=np.int32))
nvtext.token_count(strs,' ',devptr=d_arr.device_ctypes_pointer.value)
print(" ",d_arr.copy_to_host())

tokens = nvtext.unique_tokens(strs)
print("unique_tokens:",tokens)

print("contains_strings:",nvtext.contains_strings(strs,tokens))
d_arr = rmm.to_device(np.arange(strs.size()*tokens.size(),dtype=np.byte))
nvtext.contains_strings(strs,tokens,devptr=d_arr.device_ctypes_pointer.value)
print(" ",d_arr.copy_to_host())

print("strings_counts:",nvtext.strings_counts(strs,tokens))
d_arr = rmm.to_device(np.arange(strs.size()*tokens.size(),dtype=np.int32))
nvtext.strings_counts(strs,tokens,devptr=d_arr.device_ctypes_pointer.value)
print(" ",d_arr.copy_to_host())

#
print("contains_strings(cat,dog,bird,the):",nvtext.contains_strings(strs,['cat','dog','bird','the']))
print("strings_counts(cat,dog,bird,the):",nvtext.strings_counts(strs,['cat','dog','bird','the']))

#
strs = nvstrings.to_device(["kitten","kitton","kittén"])
print(strs)
print("edit_distance(kitten):",nvtext.edit_distance(strs,'kitten'))
print("edit_distance(kittén):",nvtext.edit_distance(strs,'kittén'))
strs1 = nvstrings.to_device(["kittén","sitting","Saturday","Sunday","book","back"])
print("s1:",strs1)
strs2 = nvstrings.to_device(["sitting","kitten","Sunday","Saturday","back","book"])
print("s2:",strs2)
print("edit_distance(s1,s2):",nvtext.edit_distance(strs1,strs2))
strs1 = None
strs2 = None

strs = None
tokens = None