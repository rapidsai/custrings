#
import nvstrings

#
from librmm_cffi import librmm as rmm
from librmm_cffi import librmm_config as rmm_cfg
rmm_cfg.use_pool_allocator = True 
rmm.initialize()

#
strs = nvstrings.to_device(["héllo",None,"a_bc_déf","a__bc","_ab_cd","ab_cd_",""])
print("strs:",strs)
print("strs.split(_):")
nstrs = strs.split("_")
for s in nstrs:
   print(" ",s)

print("strs.split_record(_):")
nstrs = strs.split_record("_")
for s in nstrs:
   print(" ",s)

print("strs.split(_,1):")
nstrs = strs.split("_",1)
for s in nstrs:
   print(" ",s)

print("strs.split_record(_,1):")
nstrs = strs.split_record("_",1)
for s in nstrs:
   print(" ",s)

print("strs.split(_,2):")
nstrs = strs.split("_",2)
for s in nstrs:
   print(" ",s)

print("strs.split_record(_,2):")
nstrs = strs.split_record("_",2)
for s in nstrs:
   print(" ",s)

print("strs.split(_,3):")
nstrs = strs.split("_",3)
for s in nstrs:
   print(" ",s)

print("strs.split_record(_,3):")
nstrs = strs.split_record("_",3)
for s in nstrs:
   print(" ",s)

print("strs.split(_,4):")
nstrs = strs.split("_",4)
for s in nstrs:
   print(" ",s)

print("strs.split_record(_,4):")
nstrs = strs.split_record("_",4)
for s in nstrs:
   print(" ",s)

#
print("strs.rsplit(_):")
nstrs = strs.rsplit("_")
for s in nstrs:
   print(" ",s)

print("strs.rsplit_record(_):")
nstrs = strs.rsplit_record("_")
for s in nstrs:
   print(" ",s)

print("strs.rsplit(_,1):")
nstrs = strs.rsplit("_",1)
for s in nstrs:
   print(" ",s)

print("strs.rsplit_record(_,1):")
nstrs = strs.rsplit_record("_",1)
for s in nstrs:
   print(" ",s)

print("strs.rsplit(_,2):")
nstrs = strs.rsplit("_",2)
for s in nstrs:
   print(" ",s)

print("strs.rsplit_record(_,2):")
nstrs = strs.rsplit_record("_",2)
for s in nstrs:
   print(" ",s)

print("strs.rsplit(_,3):")
nstrs = strs.rsplit("_",3)
for s in nstrs:
   print(" ",s)

print("strs.rsplit_record(_,3):")
nstrs = strs.rsplit_record("_",3)
for s in nstrs:
   print(" ",s)

print("strs.rsplit(_,4):")
nstrs = strs.rsplit("_",4)
for s in nstrs:
   print(" ",s)

print("strs.rsplit_record(_,4):")
nstrs = strs.rsplit_record("_",4)
for s in nstrs:
   print(" ",s)

#
print("strs.partition(_):")
nstrs = strs.partition('_')
for s in nstrs:
   print(" ",s)
   nvstrings.free(s)
#
print("strs.rpartition(_):")
rstrs = strs.rpartition('_')
for s in rstrs:
   print(" ",s)
   nvstrings.free(s)

strs = None
