#
import nvstrings

#
strs = nvstrings.to_device(["abc","defghi",None,"jkl","mno","pqr","stu","dog and cat","accénted",""])
print(strs)
print(".sort(1):",strs.sort(1))
print(".sort(2):",strs.sort(2))
print(".sort(2,desc):",strs.sort(2,False))
print(".sort(3):",strs.sort(3))

print(".order(1):",strs.order(1))
print(".order(2):",strs.order(2))
print(".order(2,desc):",strs.order(2,False))
print(".order(3):",strs.order(3))
