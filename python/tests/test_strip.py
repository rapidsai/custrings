#
import nvstrings

#
from librmm_cffi import librmm as rmm
from librmm_cffi import librmm_config as rmm_cfg
rmm_cfg.use_pool_allocator = True 
rmm.initialize()

#
strs = nvstrings.to_device(["  hello  ","  there  ","  world  ", None, "  accénté  ",""])
print(strs)

print(".strip():",strs.strip())
print(".lstrip():",strs.lstrip())
print(".rstrip():",strs.rstrip())

print(".strip().strip(e):",strs.strip().strip('e'))
print(".strip().strip(é):",strs.strip().strip('é'))

strs = None
