
import pandas as pd
import nvstrings
import time

strs = nvstrings.from_csv('/home/jovyan/tweets.csv', 7).to_host()

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
for i in range(50):
    #
    vlist.extend(vlist1)
    stats['strings'].append(len(vlist))
    #
    dstrs = nvstrings.to_device(vlist)
    hstrs = pd.Series(vlist)
    #
    st = time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
    d = dstrs.find("#")
    et1 = (time.clock_gettime(time.CLOCK_MONOTONIC_RAW) - st)
    print("nvstrings.find() = %05f" % et1)
    stats['nvstrings'].append(et1)
    #
    st = time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
    h = hstrs.str.find("#")
    et2 = (time.clock_gettime(time.CLOCK_MONOTONIC_RAW) - st)
    print("     pandas.find() = %05f" % et2)
    stats['pandas'].append(et2)
    print("speedup = %0.1fx" % (et2/et1) )
    #
    d = None
    h = None

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