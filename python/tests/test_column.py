#
import nvstrings

#
from librmm_cffi import librmm as rmm
from librmm_cffi import librmm_config as rmm_cfg
rmm_cfg.use_pool_allocator = True 
rmm.initialize()
#
strs = nvstrings.from_csv("../../data/7584-rows.csv",1)
#print(strs)

cols = strs.split_column(" ",2);
print(cols[1])
#print(cols[1].len())

strs = None
for c in cols:
    nvstrings.free(c)