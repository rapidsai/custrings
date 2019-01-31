import pandas as pd
import nvstrings
import time

# setup rmm to use memory pool
from librmm_cffi import librmm as rmm
from librmm_cffi import librmm_config as rmm_cfg
rmm_cfg.use_pool_allocator = True 
rmm_cfg.initial_pool_size = 2<<30  # set to 2GiB. Default is 1/2 total GPU memory
rmm.initialize()

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
da = dstrs.split(' ')
et1 = (time.clock_gettime(time.CLOCK_MONOTONIC_RAW) - st)
print("nvstrings.split() = %05f" % et1)

#
st = time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
ha = hstrs.str.split(' ')
et2 = (time.clock_gettime(time.CLOCK_MONOTONIC_RAW) - st)
print("   pandas.split() = %05f" % et2)
print("speedup = %0.5fx" % (et2/et1) )

dstrs = None
for d in da:
    nvstrings.free(d)