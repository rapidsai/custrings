#
import nvstrings

#
s1 = nvstrings.to_device(["defghi",None,"jkl","dog and cat","accénted",""])
print("s1",s1)
print("s1,s1,s1",nvstrings.from_strings(s1,s1,s1))

s2 = nvstrings.to_device(["aaa",None,"","bbb"])
print("s2",s2)
print("s1.add_strings(s2)",s1.add_strings(s2))

