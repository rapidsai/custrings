#
import nvstrings
import numpy as np

#
from librmm_cffi import librmm as rmm
from librmm_cffi import librmm_config as rmm_cfg
rmm_cfg.use_pool_allocator = True
rmm.initialize()

#
s = nvstrings.to_device(["1234","5678","90",None,"-876","543.2","-0.12",".55","-.002","","de","abc123","123abc","456e","-1.78e+5"])
print(s)
#
print(".stoi():",s.stoi())
arr = np.arange(s.size(),dtype=np.int32)
d_arr = rmm.to_device(arr)
s.stoi(d_arr.device_ctypes_pointer.value)
print(".stoi(devptr):",d_arr.copy_to_host())

#
print(".stof():",s.stof())
arr = np.arange(s.size(),dtype=np.float32)
d_arr = rmm.to_device(arr)
s.stof(d_arr.device_ctypes_pointer.value)
print(".stof(devptr):",d_arr.copy_to_host())

#
print(".hash():",s.hash())
arr = np.arange(s.size(),dtype=np.uint32)
d_arr = rmm.to_device(arr)
s.hash(d_arr.device_ctypes_pointer.value)
print(".hash(devptr):",d_arr.copy_to_host())

#
s = nvstrings.to_device(['1234567890', 'de', '1.75', '-34', '+9.8', '7¼', 'x³', '2³', '12⅝','','\t\r\n '])
print(s)
arr = np.arange(s.size(),dtype=np.byte)
d_arr = rmm.to_device(arr)

#
print(".isalnum():",s.isalnum())
s.isalnum(d_arr.device_ctypes_pointer.value)
print(".isalnum(devptr):",d_arr.copy_to_host())

#
print(".isalpha():",s.isalpha())
s.isalpha(d_arr.device_ctypes_pointer.value)
print(".isalpha(devptr):",d_arr.copy_to_host())

#
print(".isdigit():",s.isdigit())
s.isdigit(d_arr.device_ctypes_pointer.value)
print(".isdigit(devptr):",d_arr.copy_to_host())

#
print(".isdecimal():",s.isdecimal())
s.isdecimal(d_arr.device_ctypes_pointer.value)
print(".isdecimal(devptr):",d_arr.copy_to_host())

#
print(".isspace():",s.isspace())
s.isspace(d_arr.device_ctypes_pointer.value)
print(".isspace(devptr):",d_arr.copy_to_host())

#
print(".isnumeric():",s.isnumeric())
s.isnumeric(d_arr.device_ctypes_pointer.value)
print(".isnumeric(devptr):",d_arr.copy_to_host())

# htoi
s = nvstrings.to_device(["1234","ABCDEF","1A2","cafe"])
print(s)
print(".htoi()",s.htoi())
arr = np.arange(s.size(),dtype=np.uint32)
d_arr = rmm.to_device(arr)
s.htoi(d_arr.device_ctypes_pointer.value)
print(".htoi(devptr)",d_arr.copy_to_host())

# itos
print("itos():",nvstrings.itos(d_arr))
nulls = np.empty(int(s.size()/8)+1, dtype=np.int8)
nulls[0] = 11
arr = d_arr.copy_to_host()
print("itos(nulls=\\b1011):",nvstrings.itos(arr,nulls=nulls))

# ftos
arr = np.array([1.23, 123456789, 0.00000987, -12.34, 9.87e+5, 6.54e-5, np.nan, np.inf], dtype=np.float32)
d_arr = rmm.to_device(arr)
s = nvstrings.ftos(d_arr)
print("ftos:",s)
nulls = np.empty(int(s.size()/8)+1, dtype=np.int8)
nulls[0] = 11
s = nvstrings.ftos(arr,nulls=nulls)
print("ftos(nulls=\\b1011):",s)

s = nvstrings.to_device(["192.168.0.1","10.0.0.1",None,"","hello"])
print(s)
print(".ip2int()",s.ip2int())
print("int2ip()",nvstrings.int2ip(s.ip2int()))

s = nvstrings.to_device(["2019-03-20T12:34:56.789Z","2020-02-29"])
print(s)
print(".timestamp2int()",s.timestamp2int())
print(".timestamp2int(ms)",s.timestamp2int(units='milliseconds'))
print("int2timestamp()",nvstrings.int2timestamp([1553085296,1582934400]))
print("int2timestamp(ms)",nvstrings.int2timestamp([1553085296789,1582934400000],units='milliseconds'))

s = nvstrings.to_device(["true","false",None,"","true"])
print(s)
print(".to_booleans()",s.to_booleans(true="true"))
print("from_booleans",nvstrings.from_booleans([True,False,False,True],nulls=[11]))


s = None