#
import nvstrings
#
from librmm_cffi import librmm as rmm
from librmm_cffi import librmm_config as rmm_cfg
rmm_cfg.use_pool_allocator = True 
rmm.initialize()
#
strs = nvstrings.to_device(["quick brown fox jumped over lazy brown dog",None,"hello there, accéntéd world",""])
print(strs)
print(".wrap(10):",strs.wrap(10))
print(".wrap(20):",strs.wrap(20))
print(".wrap(50):",strs.wrap(50))

strs = None