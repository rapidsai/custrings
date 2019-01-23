
import nvstrings

strs = nvstrings.to_device(["hello","there","world","accéntéd",None,""])
print(strs)
print(".translate():",strs.translate([]))
print(".translate([[e,a]]):",strs.translate([['e','a']]))
print(".translate([[e,é]]):",strs.translate([['e','é']]))
print(".translate([[é,e],[o,None]]):",strs.translate([['é','e'],['o',None]]))

print(".translate(maketrans(e,a):",strs.translate(str.maketrans('e','a')))
print(".translate(maketrans(elh,ELH):",strs.translate(str.maketrans('elh','ELH')))

import string
print()
strs = nvstrings.to_device(["This, of course, is only an example!","And; will have @all the #punctuation that $money can buy.","The %percent & the *star along with the (parenthesis) with dashes-and-under_lines.","Equations: 3+3=6; 3/4 < 1 and > 0"])
print(strs)
print(".translate(punctuation=None):\n",strs.translate(str.maketrans('','',string.punctuation)))
print(".translate(punctuation=' '):\n",strs.translate(str.maketrans(string.punctuation,' '*len(string.punctuation))))