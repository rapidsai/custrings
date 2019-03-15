from librmm_cffi import librmm as rmm
from librmm_cffi import librmm_config as rmm_cfg

# setup rmm to use memory pool
rmm_cfg.use_pool_allocator = True 
rmm_cfg.initial_pool_size = 2<<30  # set to 2GiB. Default is 1/2 total GPU memory
rmm_cfg.use_managed_memory = False # default is false
rmm_cfg.enable_logging = True
rmm.initialize()

import nvstrings

#
strs = nvstrings.to_device(["Hello","there","world",None,"1234","-123.4","accénted",""])
print(strs)

# case
print(".lower():",strs.lower())
print(".upper():",strs.upper())
print(".swapcase():",strs.swapcase())
print(".capitalize():",strs.capitalize())
print(".title():",strs.title())

# combine
print(".cat([1,2,3,4,5,6,é,nil]:",strs.cat(["1","2","3","4","5","6","é",None]))
print(".join(:):",strs.join(sep=':'))

# compare
print(".compare(there):",strs.compare("there"))
print(".find(o):",strs.find('o'))
print(".rfind(e):",strs.rfind('e'))

# convert
print(".stoi():",strs.stoi())
print(".stof():",strs.stof())
print(".hash():",strs.hash())

# pad
print(".pad(8):",strs.pad(8))
print(".zfill(7):",strs.zfill(7))
print(".repeat(2):",strs.repeat(2))

# strip
print(".strip(e):",strs.strip('e'))

# slice
print(".slice(2,4):",strs.slice(2,4))
print(".slice_replace(2,4,z):",strs.slice_replace(2,4,'z'))
print(".replace(e,é):",strs.replace('e','é'))

# split
nstrs = strs.split("e")
print(".split(e):")
for s in nstrs:
   print(" ",s)
   nvstrings.free(s) # very important

nstrs = None
# this will free the strings object which deallocates from rmm
# this is important because rmm may be destroyed before the strings are
strs = None
#print(rmm.csv_log())
# not necessary here
#rmm.finalize()