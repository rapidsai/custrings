
import nvstrings
import numpy as np
values = np.array([97, 112, 112, 108, 101], dtype=np.int8)
print("values",values.tobytes())
offsets = np.array([0,1,2,3,4,5], dtype=np.int32)
print("offsets",offsets)
s = nvstrings.from_offsets(values,offsets,5)
print(s)

bitmask = np.array([29], dtype=np.int8)
print("bitmask",bitmask.tobytes())
s = nvstrings.from_offsets(values,offsets,5,bitmask,1)
print(s)

print("------------------")
values = np.array([97, 112, 112, 108, 101, 112, 101, 97, 114], dtype=np.int8)
print("values",values.tobytes())
offsets = np.array([0,5,5,9], dtype=np.int32)
print("offsets",offsets)
s = nvstrings.from_offsets(values,offsets,3)
print(s)

bitmask = np.array([5], dtype=np.int8)
print("bitmask",bitmask.tobytes())
s = nvstrings.from_offsets(values,offsets,3,bitmask,1)
print(s)

print("values.ctypes.data",hex(values.ctypes.data))
print("offsets.ctypes.data",hex(offsets.ctypes.data))
print("bitmask.ctypes.data",hex(bitmask.ctypes.data))
s = nvstrings.from_offsets(values.ctypes.data,offsets.ctypes.data,3,bitmask.ctypes.data,1)
print(s)

print("------------------")
s = nvstrings.to_device(['a','p','p','l','e'])
values = np.empty(s.size(), dtype=np.int8)
offsets = np.empty(s.size()+1, dtype=np.int32)
nulls = np.empty(int(s.size()/8)+1, dtype=np.int32)
s.to_offsets(values,offsets,nulls)
print("values",values.tobytes())
print("offsets",offsets)
print("nulls",nulls.tobytes())

print("------------------")
import nvcategory

values = np.array([97, 112, 112, 108, 101], dtype=np.int8)
print("values",values.tobytes())
offsets = np.array([0,1,2,3,4,5], dtype=np.int32)
print("offsets",offsets)
c = nvcategory.from_offsets(values,offsets,5)
print(c.keys(),c.values())

bitmask = np.array([29], dtype=np.int8)
print("bitmask",bitmask.tobytes())
c = nvcategory.from_offsets(values,offsets,5,bitmask,1)
print(c.keys(),c.values())