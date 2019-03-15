import nvstrings

strs = nvstrings.to_device(['', None, 'a b', ' a b ', '  aa  bb  ', ' a  bbb   c', ' aa b  ccc  '])

print("split_record():")
for s in strs.split_record():
    print(" ",s)

print("split_record(n=1):")
for s in strs.split_record(n=1):
    print(" ",s)

print("split_record(n=2):")
for s in strs.split_record(n=2):
    print(" ",s)

print("rsplit_record():")
for s in strs.rsplit_record():
    print(" ",s)

print("rsplit_record(n=1):")
for s in strs.rsplit_record(n=1):
    print(" ",s)

print("rsplit_record(n=2):")
for s in strs.rsplit_record(n=2):
    print(" ",s)

print("split():")
for s in strs.split():
    print(" ",s)

print("split(n=1):")
for s in strs.split(n=1):
    print(" ",s)

print("split(n=2):")
for s in strs.split(n=2):
    print(" ",s)

print("rsplit():")
for s in strs.rsplit():
    print(" ",s)

print("rsplit(n=1):")
for s in strs.rsplit(n=1):
    print(" ",s)

print("rsplit(n=2):")
for s in strs.rsplit(n=2):
    print(" ",s)

strs = nvstrings.to_device(['ab', 'c', 'd', 'e', 'f'])
print(strs)
for s in strs.split():
    print(" ",s)
for s in strs.rsplit():
    print(" ",s)    