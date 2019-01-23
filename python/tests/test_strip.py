#
import nvstrings

#
strs = nvstrings.to_device(["  hello  ","  there  ","  world  ", None, "  accénté  ",""])
print(strs)

print(".strip():",strs.strip())
print(".lstrip():",strs.lstrip())
print(".rstrip():",strs.rstrip())

print(".strip().strip(e):",strs.strip().strip('e'))
print(".strip().strip(é):",strs.strip().strip('é'))
