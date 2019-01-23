#
import nvstrings

#
strs = nvstrings.from_csv("../../data/7584-rows.csv",1)
#print(strs)

cols = strs.split_column(" ",2);
print(cols[1])
#print(cols[1].len())