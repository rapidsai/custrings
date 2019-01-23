#
import nvstrings

#
strs = nvstrings.to_device(["abc","Def",None,"jLl","mnO","PqR","sTT","dog and cat","accénted",""," 1234 ","XYZ"])
print(strs)
print(".lower():",strs.lower())
print(".upper():",strs.upper())
print(".swapcase():",strs.swapcase())
print(".capitalize():",strs.capitalize())
print(".title():",strs.title())
# only first char (pos=0) is capitalized -- not the first letter
print(".rjust(4).capitalize():",strs.rjust(4).capitalize())

#
print(".islower():",strs.islower())
print(".isupper():",strs.isupper())