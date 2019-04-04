import nvstrings
import pickle

filehandler = open("/tmp/ipctest", 'rb') 
ipc_data = pickle.load(filehandler)
#print(ipc_data)

new_strs = nvstrings.create_from_ipc(ipc_data)
print(new_strs)
