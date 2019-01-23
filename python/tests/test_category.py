#
import nvstrings, nvcategory

# create
strs = nvstrings.to_device(["eee","aaa","eee","ddd","ccc","ccc","ccc","eee","aaa"])
print(strs.size(),strs)
cat = nvcategory.from_strings(strs)
print(cat.size(),cat)

print(".values():",cat.values())
print(".value_for_index(7)",cat.value_for_index(7))
print(".value(ccc):",cat.value('ccc'))
print(".indexes_for_key(ccc):",cat.indexes_for_key('ccc'))
print(".to_strings():",cat.to_strings())

# add
print("-------------------------")
print("add strings:")
strs = nvstrings.to_device(["ggg","fff","hhh","aaa","fff","fff","ggg","hhh","bbb"])
print(strs.size(),strs)
cat = cat.add_strings(strs)
print(cat.size(),cat.keys())

print(".values():",cat.values())
print(".value_for_index(7)",cat.value_for_index(7))
print(".value(aaa):",cat.value('aaa'))
print(".indexes_for_key(aaa):",cat.indexes_for_key('aaa'))
print(".to_strings():",cat.to_strings())
print(".gather_strings([0,2,0]):",cat.gather_strings([0,2,0]))

# remove
print("-------------------------")
print("remove strings:")
strs = nvstrings.to_device(["ccc","aaa","bbb"])
print(strs.size(),strs)
cat = cat.remove_strings(strs)
print(cat.size(),cat.keys())

print(".values():",cat.values())
print(".value_for_index(7)",cat.value_for_index(7))
print(".value(fff):",cat.value('fff'))
print(".indexes_for_key(fff):",cat.indexes_for_key('fff'))
print(".to_strings():",cat.to_strings())

# multiple strings in one call
print("-------------------------")
strs1 = nvstrings.to_device(["eee","aaa","eee","ddd","ccc","ccc","ccc","eee","aaa"])
strs2 = nvstrings.to_device(["ggg","fff","hhh","aaa","fff","fff","ggg","hhh","bbb"])
print(".from_strings(strs1,strs2)")
cat = nvcategory.from_strings(strs1,strs2)
print(cat.size(),cat)

print(".values():",cat.values())
print(".value(ccc):",cat.value('ccc'))
print(".indexes_for_key(ccc):",cat.indexes_for_key('ccc'))
print(".gather_strings([0,2,0,3,1]):",cat.gather_strings([0,2,0,3,1]))

# Masonry, Reinforced Concrete, Reinforced Masonry, Steel Frame, Wood
print("-------------------------")
print("36634-rows.csv:")
strs = nvstrings.from_csv("../../data/36634-rows.csv",16)
cat = nvcategory.from_strings(strs)
print(cat.size(),cat.keys())
print("len(.values()):",len(cat.values()))
print(".value(Wood):",cat.value('Wood'))