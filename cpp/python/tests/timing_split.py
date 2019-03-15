import pandas as pd
import nvstrings
import time

df = pd.read_csv('/data/7584-rows.csv', sep=',')
df.columns

values = df["address"].values
values

dstrs = nvstrings.to_device(values.tolist())
hstrs = pd.Series(values.tolist())

print("precision = %0.9f seconds" % time.clock_getres(time.CLOCK_MONOTONIC_RAW))
print(str(dstrs.size()), "strings")
#
st = time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
d = dstrs.split(' ')
et1 = (time.clock_gettime(time.CLOCK_MONOTONIC_RAW) - st)
print("nvstrings.split() = %05f" % et1)

#
st = time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
h = hstrs.str.split(' ')
et2 = (time.clock_gettime(time.CLOCK_MONOTONIC_RAW) - st)
print("     pandas.split() = %05f" % et2)
print("speedup = %0.5fx" % (et2/et1) )

# clear output
d = None
h = None

