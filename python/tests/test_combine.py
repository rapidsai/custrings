#
import nvstrings

#
from librmm_cffi import librmm as rmm
from librmm_cffi import librmm_config as rmm_cfg
rmm_cfg.use_pool_allocator = True 
rmm.initialize()

#
strs1 = nvstrings.to_device(["abc","def",None,"","jkl","mno","accént"])
print("strs1:",strs1)
print(".cat():",strs1.cat())
print(".cat(sep=:):",strs1.cat(sep=":"))
print(".cat(sep=:,na_rep=_):",strs1.cat(sep=":",na_rep="_"))
print(".cat([1,2,3,4,5,é,nil],sep=:,na_rep=_):",strs1.cat(["1","2","3","4","5","é",None],sep=":",na_rep="_"))

strs2 = nvstrings.to_device(["1","2","3",None,"5","é",""])
print("strs2:",strs2)
print("strs1.cat(strs2):",strs1.cat(strs2))
print("strs1.cat(strs2,sep=:):",strs1.cat(strs2,sep=":"))
print("strs1.cat(strs2,sep=:,na_rep=_):",strs1.cat(strs2,sep=":",na_rep="_"))

print("strs1.join():",strs1.join())
print("strs1.join(sep=:):",strs1.join(sep=":"))

strs1 = None
strs2 = None