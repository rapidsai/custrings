import nvstrings
import time

#input = ['abcxyz']*10000000
input = ['u']*10000000
st = time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
nvstrings.to_device(input)
et1 = (time.clock_gettime(time.CLOCK_MONOTONIC_RAW) - st)
print("nvstrings.to_device() = %05f" % et1)

#
