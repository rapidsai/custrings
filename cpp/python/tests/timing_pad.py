import pandas as pd
import nvstrings
import time

df = pd.read_csv('/home/jovyan/7584-rows.csv', sep=',')
df.columns

values = df["address"].values
values

vlist = values.tolist()
vlist.extend(vlist)
vlist.extend(vlist)

print("precision = %0.9f seconds" % time.clock_getres(time.CLOCK_MONOTONIC_RAW))

for i in range(5):
    vlist.extend(vlist)
    len(vlist)
    #
    dstrs = nvstrings.to_device(vlist)
    hstrs = pd.Series(vlist)
    #
    st = time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
    d = dstrs.pad(30)
    et1 = (time.clock_gettime(time.CLOCK_MONOTONIC_RAW) - st)
    print("nvstrings.pad() = %05f" % et1)
    #
    st = time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
    h = hstrs.str.pad(30)
    et2 = (time.clock_gettime(time.CLOCK_MONOTONIC_RAW) - st)
    print("     pandas.pad() = %05f" % et2)
    print("speedup = %0.1fx" % (et2/et1) )
    #
    d = None
    h = None
