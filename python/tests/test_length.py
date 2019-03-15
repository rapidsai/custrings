#
import nvstrings
from librmm_cffi import librmm as rmm

#
strs = nvstrings.to_device(["abc","Def",None,"jLl","mnO","PqR","sTT","dog and cat","accénted",""," 1234 ","XYZ"])
print(strs)
#
print("len(strs):",len(strs))
print(".size():",strs.size())
print(".len():",strs.len())
print(".byte_count():",strs.byte_count())
print(".null_count():",strs.null_count())

