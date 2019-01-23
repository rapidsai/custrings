
import pandas as pd
import nvstrings
import time

#df = pd.read_csv('/home/jovyan/reviews-1m.csv', sep=',')
#values = df["text"].values
#vlist = values.tolist()

print("precision = %0.9f seconds" % time.clock_getres(time.CLOCK_MONOTONIC_RAW))

for i in range(3):
    lines = (i+1) * 1000000
    #vlist.extend(vlist)
    #print("strings:",len(vlist))
    #
    #dstrs = nvstrings.to_device(vlist)
    dstrs = nvstrings.from_csv("/home/jovyan/reviews.txt",0,lines=lines)
    vlist = dstrs.to_host()
    print("strings = ",len(vlist))
    hstrs = pd.Series(vlist)
    #
    st = time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
    d = dstrs.slice(3,103)
    et1 = (time.clock_gettime(time.CLOCK_MONOTONIC_RAW) - st)
    print("nvstrings.slice() = %05f" % et1)
    #
    st = time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
    h = hstrs.str.slice(3,103)
    et2 = (time.clock_gettime(time.CLOCK_MONOTONIC_RAW) - st)
    print("   pandas.slice() = %05f" % et2)
    print("speedup = %0.1fx" % (et2/et1) )
    #
    d = None
    h = None

