import nvstrings
import pickle

strs = nvstrings.to_device(["abc","defghi",None,"jkl","mno","pqr","stu","dog and cat","accénted",""])
print(strs)

ipc_data = strs.get_ipc_data()
#print(ipc_data)

with open("/tmp/ipctest", 'wb') as filehandler:
    pickle.dump(ipc_data, filehandler)

input("Press Enter to continue...")