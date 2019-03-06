#
import nvstrings
#
from librmm_cffi import librmm as rmm
from librmm_cffi import librmm_config as rmm_cfg
rmm_cfg.use_pool_allocator = True 
rmm.initialize()

#
strs = nvstrings.to_device(["5","hej","\t \n","12345","\\","d","c:\\Tools","+27", "1c2", "1C2" ])
print(strs)
print(".contains('\\d')", strs.contains('\\d') )
print(".contains('\\w+')", strs.contains('\\w+') )
print(".contains('\\s')", strs.contains('\\s') )
print(".contains('\\S')", strs.contains('\\S') )
print(".contains('^.*\\\\.*$')", strs.contains('^.*\\\\.*$') )

print("----------------------")
strs2 = nvstrings.to_device([ "0123456789", "1C2", "Xaa", "abcdefghxxx", "ABCDEFGH", "abcdefgh", "abc def", "abc\ndef", "aa\r\nbb\r\ncc\r\n\r\n", "abcabc" ])
print(strs2)
print(".contains('[1-5]+')", strs2.contains('[1-5]+') )
print(".contains('[a-h]+')", strs2.contains('[a-h]+') )
print(".contains('[A-H]+')", strs2.contains('[A-H]+') )
print(".contains('\\n')", strs2.contains('\n') )
print(".contains('b.\\s*\\n')", strs2.contains('b.\\s*\n') )
print(".contains('.*c')", strs2.contains('.*c') )

print("----------------------")
strs3 = nvstrings.to_device([  "0:00:0", "0:0:00", "00:0:0", "00:00:0", "00:0:00", "0:00:00", "00:00:00", "Hello world !", "Hello world!   ", "Hello worldcup  !" ])
print(strs3)
print(".contains('[0-9]')", strs3.contains('[0-9]') )
print(".contains('\\d\\d:\\d\\d:\\d\\d')", strs3.contains('\\d\\d:\\d\\d:\\d\\d') )
print(".contains('\\d\\d?:\\d\\d?:\\d\\d?')", strs3.contains('\\d\\d?:\\d\\d?:\\d\\d?') )
print(".contains('[Hh]ello [Ww]orld')", strs3.contains('[Hh]ello [Ww]orld') )
print(".contains('\\bworld\\b')", strs3.contains('\\bworld\\b') )

print("----------------------")
strs4 =  nvstrings.to_device(["hello @abc @def world", "The quick brown @fox jumps", "over the", "lazy @dog", "hello http://www.world.com I'm here @home"])
print(strs4)
print(".replace(@\\S+,***):\n  ",strs4.replace("@\\S+","***"))
print(".replace((?:@|https?://)\\S+,''):\n  ",strs4.replace("(?:@|https?://)\\S+",""))

print("----------------------")
strs = nvstrings.to_device(["hello","and héllo",None,""])
print(strs)
print(".contains('h[a-u]llo')", strs.contains('h[a-u]llo'))
print(".contains('h[á-ú]llo')", strs.contains('h[á-ú]llo'))
print(".contains('é',False)", strs.contains('é',False))
print(".match('[hH]')", strs.match('[hH]'))

print("----------------------")
strs = nvstrings.to_device(["A","B","Aaba","Baca",None,"CABA","cat",""])
print(strs)
print(".count(a):", strs.count('a'))
print(".count([aA]):", strs.count('[aA]'))
print(".match('[bB][aA]'):", strs.match('[bB][aA]'))
print(".findall('[aA]'):")
columns = strs.findall('[aA]')
for col in columns:
	print(" ",col)
print(".findall_record('[aA]'):")
rows = strs.findall_record('[aA]')
for row in rows:
	print(" ",row)

print("----------------------")
strs = nvstrings.to_device(['ALA-PEK Flight:HU7934', 'HKT-PEK Flight:CA822', 'FRA-PEK Flight:LA8769', 'FRA-PEK Flight:LH7332', '', None, 'Flight:ZZ'])
print(strs)
print(".extract(r'Flight:([A-Z]+)(\d+)'):")
columns = strs.extract(r'Flight:([A-Z]+)(\d+)')
for col in columns:
	print(" ",col)
	nvstrings.free(col)
print(".extract_record(r'Flight:([A-Z]+)(\d+)'):")
rows = strs.extract_record(r'Flight:([A-Z]+)(\d+)')
for row in rows:
	print(" ",row)
	nvstrings.free(row)

print("----------------------")
strs = nvstrings.to_device('word [[wikt:anarchism|anarchism]] is')
print(strs)
print(".replace('\\[\\[[a-z\\-]+:[^]]+\\]\\]','')",strs.replace('\\[\\[[a-z\\-]+:[^]]+\\]\\]',''))

print("----------------------")
strs = nvstrings.to_device(["A543","Z756","",None])
print(strs)
print("(\\d)(\\d),'\\1-\\2'",strs.replace_with_backrefs('(\\d)(\\d)', '\\1-\\2'))
print("(\\d)(\\d),'V\\2-\\1'",strs.replace_with_backrefs('(\\d)(\\d)', 'V\\2-\\1'))
print("(\\d)(\\d),'V\\1-\\3'",strs.replace_with_backrefs('(\\d)(\\d)', 'V\\1-\\3'))
print("(\\d)(\\d),'V\\3-\\2'",strs.replace_with_backrefs('(\\d)(\\d)', 'V\\3-\\2'))

strs = None
strs1 = None
strs2 = None
strs3 = None
strs4 = None
