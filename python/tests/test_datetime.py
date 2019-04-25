#
import nvstrings
import numpy as np

#
from librmm_cffi import librmm as rmm
from librmm_cffi import librmm_config as rmm_cfg
rmm_cfg.use_pool_allocator = True
rmm.initialize()

s = nvstrings.to_device(["2019-03-20T12:34:56Z","2020-02-29T23:59:59Z"])
print(s)
print(".timestamp2int()",s.timestamp2int())
print(".timestamp2int(ms)",s.timestamp2int(units='ms'))

s = nvstrings.to_device(["04/24/2019","02/29/2020"])
print(".timestamp2int(M/D/Y)",s.timestamp2int(format='%m/%d/%Y'))
print(".timestamp2int(M/D/Y,Y)",s.timestamp2int(format='%m/%d/%Y',units='Y'))
print(".timestamp2int(M/D/Y,M)",s.timestamp2int(format='%m/%d/%Y',units='M'))
print(".timestamp2int(M/D/Y,D)",s.timestamp2int(format='%m/%d/%Y',units='D'))

s = nvstrings.to_device(["04/24/2019 01:23:45","02/29/2020 12:34:56"])
print(".timestamp2int(M/D/Y H:M:S)",s.timestamp2int(format='%m/%d/%Y %H:%M:%S'))
print(".timestamp2int(M/D/Y H:M:S,h)",s.timestamp2int(format='%m/%d/%Y %H:%M:%S',units='h'))
print(".timestamp2int(M/D/Y H:M:S,m)",s.timestamp2int(format='%m/%d/%Y %H:%M:%S',units='m'))
print(".timestamp2int(M/D/Y H:M:S,s)",s.timestamp2int(format='%m/%d/%Y %H:%M:%S',units='s'))

print("int2timestamp()",nvstrings.int2timestamp([1553085296,1582934400]))
print("int2timestamp(ms)",nvstrings.int2timestamp([1553085296789,1582934400000],format="%Y-%m-%d %H:%M:%S.%f",units='ms'))

s = None