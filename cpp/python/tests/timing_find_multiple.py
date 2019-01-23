
import nvstrings
import numpy as np
from numba import cuda
import time

print("precision = %0.9f seconds" % time.clock_getres(time.CLOCK_MONOTONIC_RAW))

lines = 1000000
dstrs = nvstrings.from_csv("/home/dwendt/data/reviews/reviews.txt",0,lines=lines)
#
# there are 14 of these:
rwords = ["fruit","vintage","zest","foam","sweet","juic","malt","wheat","citrus","pine","crisp","dark","golden","bitter"]
# there are 133 of these:
swords =['i','me','my','myself','we','our','ours','ourselves','you','your','yours','yourself',
            'yourselves','he','him','his','himself','she','her','hers','herself','it','its','itself',
            'they','them','their','theirs','themselves','what','which','who','whom','this','that',
            'these','those','am','is','are','was','were','be','been','being','have','has','had',
            'having','do','does','did','doing','a','an','the','and','but','if','or','because','as',
            'until','while','of','at','by','for','with','about','against','between','into','through',
            'during','before','after','above','below','to','from','up','down','in','out','on','off',
            'over','under','again','further','then','once','here','there','when','where','why','how',
            'all','any','both','each','few','more','most','other','some','such','no','nor','not',
            'only','own','same','so','than','too','very','s','t','can','will','just','don','should',
            'now','uses','use','using','used','one','also']

stats = []
for i in range(len(rwords)):
    #
    words = rwords[0:i+1]
    print(words)
    tgts = nvstrings.to_device(words)
    #
    arr = np.arange(tgts.size()*dstrs.size(),dtype=np.int32)
    d_arr = cuda.to_device(arr)
    devmem = d_arr.device_ctypes_pointer.value
    #
    vp = devmem
    st = time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
    for w in words:
        dstrs.find(w,devptr=vp)
        vp = vp + (tgts.size()*4)
    et1 = (time.clock_gettime(time.CLOCK_MONOTONIC_RAW) - st)
    print("         find() = %05f" % et1)
    #
    st = time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
    d = dstrs.find_multiple(words,devptr=devmem)
    et2 = (time.clock_gettime(time.CLOCK_MONOTONIC_RAW) - st)
    print("find_multiple() = %05f" % et2)
    print("speedup = %0.2fx" % (et1/et2) )
    stats.append((et1/et2))
    #

for s in stats:
    print(s,end=',')
print()
