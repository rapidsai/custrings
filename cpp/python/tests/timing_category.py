
import nvstrings, nvcategory
import numpy as np
from numba import cuda
import time

print("precision = %0.9f seconds" % time.clock_getres(time.CLOCK_MONOTONIC_RAW))

lines = 1000000
# column 5 = style (e.g. American Pale Ale, Vienna Lager, etc)
dstrs = nvstrings.from_csv("/home/dwendt/data/reviews/beers-1m.csv",5,lines=lines)
#input("press enter")
#
slist = []
stats1 = []
stats2 = []

for i in range(50):
    idx = i + 1
    #print(idx,'million')
    #
    slist.append(dstrs)
    #
    st = time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
    cat = nvcategory.from_strings_list(slist)
    et = (time.clock_gettime(time.CLOCK_MONOTONIC_RAW) - st)
    print(cat.keys_size(),cat.size())
    print("  from_strings_list = %05f" % et)
    stats1.append(et)
    #
    arr = np.arange(dstrs.size()*idx,dtype=np.int32)
    d_arr = cuda.to_device(arr)
    devmem = d_arr.device_ctypes_pointer.value
    st = time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
    vp = cat.indexes_for_key('American Lager',devmem)
    et = (time.clock_gettime(time.CLOCK_MONOTONIC_RAW) - st)
    print("  cat.indexes_for_key = %05f" % et)
    stats2.append(et)

#input("press enter")
for s in stats1:
    print(s,end=',')
print()
for s in stats2:
    print(s,end=',')
print()