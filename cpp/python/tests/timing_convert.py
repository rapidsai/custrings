
import nvstrings
import time
import numpy as np
from numba import cuda

arr = np.arange(7854)
vlist = arr.astype('str').tolist()

vlist.extend(vlist)
vlist.extend(vlist)

print("precision = %0.9f seconds" % time.clock_getres(time.CLOCK_MONOTONIC_RAW))

for i in range(9):
    vlist.extend(vlist)
    len(vlist)
    #
    dstrs = nvstrings.to_device(vlist)
    arr = np.arange(dstrs.size(),dtype=np.int32)
    d_arr = cuda.to_device(arr)
    #
    st = time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
    d = dstrs.stoi(d_arr.device_ctypes_pointer.value)
    et1 = (time.clock_gettime(time.CLOCK_MONOTONIC_RAW) - st)
    print("nvstrings.stoi() = %05f" % et1)
    #
    d = None

