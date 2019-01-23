#
import nvstrings
#
strs = nvstrings.to_device(["quick brown fox jumped over lazy brown dog",None,"hello there, accéntéd world",""])
print(strs)
print(".wrap(10):",strs.wrap(10))
print(".wrap(20):",strs.wrap(20))
print(".wrap(50):",strs.wrap(50))
