
import pandas as pd
import nvstrings
import time

dstrs_in = nvstrings.from_csv('/data/tweets.csv', 7)
vlist = dstrs_in.to_host()
vlist.extend(vlist)
vlist.extend(vlist)
vlist.extend(vlist)
vlist.extend(vlist)
vlist.extend(vlist)
vlist.extend(vlist)
vlist.extend(vlist)
vlist.extend(vlist)
len(vlist)

dstrs = nvstrings.to_device(vlist)
hstrs = pd.Series(vlist)

print("precision = %0.9f seconds" % time.clock_getres(time.CLOCK_MONOTONIC_RAW))
print("strings =",dstrs.size())
#
st = time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
d = dstrs.contains('@.+@')
et1 = (time.clock_gettime(time.CLOCK_MONOTONIC_RAW) - st)
print("nvstrings.contains('@.+@') = %05f" % et1)

st = time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
h = hstrs.str.contains('@.+@')
et2 = (time.clock_gettime(time.CLOCK_MONOTONIC_RAW) - st)
print("pandas.contains('@.+@') = %05f" % et2)
print("speedup = %0.1fx" % (et2/et1) )

#
st = time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
d = dstrs.contains('world')
et1 = (time.clock_gettime(time.CLOCK_MONOTONIC_RAW) - st)
print("nvstrings.contains('world') = %05f" % et1)

st = time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
h = hstrs.str.contains('world')
et2 = (time.clock_gettime(time.CLOCK_MONOTONIC_RAW) - st)
print("pandas.contains('world') = %05f" % et2)
print("speedup = %0.1fx" % (et2/et1) )

# 
d = None
h = None






