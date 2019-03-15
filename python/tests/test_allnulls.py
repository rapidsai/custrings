
import nvstrings

#
strs = nvstrings.to_device([None,None,None,None,None,None,None,None])
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
print(".replace(e,_):",strs.replace('e','_'))

# order/sort
print(".sort(2):",strs.sort(2))
print(".order(2):",strs.order(2))

# split
nstrs = strs.split("e")
print(".split(e):")
for s in nstrs:
   print(" ",s)
