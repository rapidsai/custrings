
import pandas as pd
import nvstrings
import time

# setup rmm to use memory pool
from librmm_cffi import librmm as rmm
from librmm_cffi import librmm_config as rmm_cfg
rmm_cfg.use_pool_allocator = True 
rmm_cfg.initial_pool_size = 8<<30  # 8GB
rmm.initialize()

strs = nvstrings.from_csv('/data/tweets.csv', 7).to_host()

vlist1 = []
vlist1.extend(strs)
vlist1.extend(strs)
vlist1.extend(strs)
vlist1.extend(vlist1)
vlist1.extend(vlist1)
vlist1.extend(vlist1)
vlist1.extend(vlist1)
vlist1.extend(vlist1)
vlist1.extend(vlist1)
vlist1.extend(vlist1)

print("precision = %0.9f seconds" % time.clock_getres(time.CLOCK_MONOTONIC_RAW))

stats = {'strings':[], 'pandas':[], 'nvstrings':[]}
vlist = []
for i in range(20):
    print(i,"--------------------")
    #
    vlist.extend(vlist1)
    stats['strings'].append(len(vlist))
    print(str(len(vlist)),"strings")
    #
    dstrs = nvstrings.to_device(vlist)
    hstrs = pd.Series(vlist)
    #
    st = time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
    d = dstrs.slice(1,15)
    et1 = (time.clock_gettime(time.CLOCK_MONOTONIC_RAW) - st)
    print("nvstrings.slice() = %05f" % et1)
    stats['nvstrings'].append(et1)
    #
    st = time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
    h = hstrs.str.slice(1,15)
    et2 = (time.clock_gettime(time.CLOCK_MONOTONIC_RAW) - st)
    print("   pandas.slice() = %05f" % et2)
    stats['pandas'].append(et2)
    print("speedup = %0.1fx" % (et2/et1) )
    #
    d = None
    h = None
    dstrs = None

line = "' '"
for n in stats['strings']:
    line = line + ',' + str(n)
print(line)

line = 'pandas'
for n in stats['pandas']:
    line = line + ',' + str(n)
print(line)

line = 'nvstrings'
for n in stats['nvstrings']:
    line = line + ',' + str(n)
print(line)