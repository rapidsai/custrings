
import pandas as pd
import nvstrings
import time

df = pd.read_csv('/home/jovyan/7584-rows.csv', sep=',')
df.columns

#values1 = df["construction"].values
values1 = df["address"].values
#values2 = df["county"].values
values2 = df["crimedescr"].values

vlist1 = values1.tolist()
vlist1.extend(vlist1)
vlist1.extend(vlist1)
vlist2 = values2.tolist()
vlist2.extend(vlist2)
vlist2.extend(vlist2)

print("precision = %0.9f seconds" % time.clock_getres(time.CLOCK_MONOTONIC_RAW))

for i in range(9):
    vlist1.extend(vlist1)
    vlist2.extend(vlist2)
    len(vlist2)
    #
    dstrs1 = nvstrings.to_device(vlist1)
    dstrs2 = nvstrings.to_device(vlist2)
    hstrs1 = pd.Series(vlist1)
    hstrs2 = pd.Series(vlist2)
    #
    st = time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
    d = dstrs1.cat(dstrs2)
    et1 = (time.clock_gettime(time.CLOCK_MONOTONIC_RAW) - st)
    print("nvstrings.cat() = %05f" % et1)
    #
    st = time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
    h = hstrs1.str.cat(hstrs2)
    et2 = (time.clock_gettime(time.CLOCK_MONOTONIC_RAW) - st)
    print("     pandas.cat() = %05f" % et2)
    print("speedup = %0.1fx" % (et2/et1) )
    #
    d = None
    h = None
