#
import nvstrings

#
from librmm_cffi import librmm as rmm
from librmm_cffi import librmm_config as rmm_cfg
rmm_cfg.use_pool_allocator = True 
rmm.initialize()
#
strs = nvstrings.to_device(["abcdefghij","0123456789","9876543210", None, "accénted", ""])
print(strs)

print(".slice(2,8):",strs.slice(2,8))
print(".slice(2,15):",strs.slice(2,15))
print(".slice(2,8,2):",strs.slice(2,8,2))
print(".slice(2,8,5):",strs.slice(2,8,5))

print(".slice_replace(2,5,z):",strs.slice_replace(2,5,'z'))
print(".slice_replace(8,8,z):",strs.slice_replace(8,8,'z'))

print(".get(0):",strs.get(0))
print(".get(3):",strs.get(3))
print(".get(9):",strs.get(9))
print(".get(10):",strs.get(10))

print(".replace(3,_):",strs.replace('3','_'))
print(".replace(3,++):",strs.replace('3','++'))
print(".replace(c,):",strs.replace('c',''))

print(".fillna(''):",strs.fillna(''))
repl = nvstrings.to_device(["null1","null2","null3","null4","null5","null6"])
print(".fillna(nvs):",strs.fillna(repl))

strs = None
repl = None
