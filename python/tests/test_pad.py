#
import nvstrings

#
from librmm_cffi import librmm as rmm
from librmm_cffi import librmm_config as rmm_cfg
rmm_cfg.use_pool_allocator = True 
rmm.initialize()
#
strs = nvstrings.to_device(["hello","there","world","1234","-1234",None,"accént",""])
print(strs)

print(".pad(5):",strs.pad(5))
print(".pad(7,right):",strs.pad(7,'right'))
print(".pad(9,both,.):",strs.pad(9,'both','.'))

print(".ljust(7):",strs.ljust(7))
print(".rjust(10):",strs.rjust(10))
print(".center(10):",strs.center(10))
print(".zfill(6):",strs.zfill(6))

print(".repeat(3):",strs.repeat(3))

strs = None