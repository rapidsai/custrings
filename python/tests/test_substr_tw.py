
import nvstrings

strs = nvstrings.from_csv('../../data/tweets.csv', 7)

print("slice(1,15):",strs.slice(1,15))
