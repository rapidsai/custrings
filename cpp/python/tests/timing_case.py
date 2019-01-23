
import pandas as pd
import nvstrings
import time

df = pd.read_csv('/home/jovyan/reviews-1m.csv', sep=',')
values = df["text"].values;

#vlist.extend(vlist)

print("precision = %0.9f seconds" % time.clock_getres(time.CLOCK_MONOTONIC_RAW))

for i in range(6):
    vlist.extend(vlist)
    print("strings:",len(vlist))
    #
    dstrs = nvstrings.to_device(vlist)
    hstrs = pd.Series(vlist)
    #
    st = time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
    d = dstrs.lower()
    et1 = (time.clock_gettime(time.CLOCK_MONOTONIC_RAW) - st)
    print("nvstrings.lower() = %05f" % et1)
    #
    st = time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
    h = hstrs.str.lower()
    et2 = (time.clock_gettime(time.CLOCK_MONOTONIC_RAW) - st)
    print("   pandas.lower() = %05f" % et2)
    print("speedup = %0.1fx" % (et2/et1) )
    #
    d = None
    h = None

