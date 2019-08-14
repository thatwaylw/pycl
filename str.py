#coding:utf-8
'''
Created on 2017年6月16日
@author: laiwei
'''
# import datetime
from datetime import datetime
#nowTime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
dt = datetime.now()
dd = dt.strftime('%Y%m%d')
tt = dt.strftime('%H%M%S')
print(dd,tt)
dd = dt.strftime('%d-%b-%y')
tt = dt.strftime('%I:%M:%S%p')
print(dd,tt)
text = '2012-09-20'
y = datetime.strptime(text, '%Y-%m-%d')
print(y)
text = '20-Jul-2019 06:49'
y = datetime.strptime(text, '%d-%b-%Y %H:%M')
print(y)

import re
# def addLessonName(txt):
#     re1 = re.compile(r'^(\(\d+\))(.+)$')
#     mt = re1.match(txt)
#     if(mt):
#         print(mt.group())
#         print(mt.group()[1])
#         print(mt.group()[4])
#         num = int(mt.group()[0])
#         name = mt.group()[1]
#         return ('(%d)%s' % (num+1, name))
#     else:
#         return ('(2)%s' % txt)
'''    
def addLessonName(txt):
    if(txt[0:3]=='(2)'):
        return '(3)'+txt[3:]
    if(txt[0:3]=='(3)'):
        return '(4)'+txt[3:]
    if(txt[0:3]=='(4)'):
        return '(5)'+txt[3:]
    if(txt[0:3]=='(5)'):
        return '(6)'+txt[3:]
    if(txt[0:3]=='(6)'):
        return '(7)'+txt[3:]
    if(txt[0:3]=='(7)'):
        return '(8)'+txt[3:]
    if(txt[0:3]=='(8)'):
        return '(9)'+txt[3:]
    else:
        return '(2)' + txt
        
print(addLessonName(' shu self assessment'))
'''
re_Unit = re.compile(r'Unit\s+(\d+)(\s+\w+)?')
str = '可以  Unit 3 大家好 哈哈 Unit 4 大 a'
print(re_Unit.findall(str))
# mt = re_Unit.match(str)     # 头尾不能有其他字符。。。
mt = re_Unit.search(str)     # 头尾不能有其他字符。。。
print(True if mt else False)
print(mt.string)
print('group(): ', mt.group())
print('group(0): ', mt.group(0))
print('group(1): ', mt.group(1))
print('group(2): ', mt.group(2))
# print('group(3): ', mt.group(3))    # 出错
print('group()[0]: ', mt.groups()[0])
print('group()[1]: ', mt.groups()[1])
# print('group()[2]: ', mt.groups()[2])   #出错！

for m in re_Unit.finditer(str):
    print('%02d-%02d: %s' % (m.start(), m.end(), m.group(0)))

text = "He was carefully disguised but captured quickly by police."
for m in re.finditer(r"\w+ly", text):
    print('%02d-%02d: %s' % (m.start(), m.end(), m.group(0)))
# 07-16: carefully
# 40-47: quickly

# str = 'abC10'
# print(str.isalpha())

# str = 'abc,\n'
# print(str[-2:])
# print(str[-2:]==',\n')
# print(str[:-2])
# print(str[:-2]+'\na')
# str = 'aaa\t\tbbb'
# tok = str.split('\t\t')
# print(tok)
# print(len(tok))

# stren = "Where's the green pencil, Green pencil, green pencil? Where's the green pencil? It's under my desk.\nWhere's the red apple, Red apple, red apple? Where's the red apple? It's on my chair."
# strcn = "绿色的铅笔在哪里，\n绿色的铅笔，绿色的铅笔？\n绿色的铅笔在哪里？\n它在我的书桌下。\n\n红色的苹果在哪里，\n红色的苹果，红色的苹果？\n红色的苹果在哪里？\n它在我的椅子上。"
#stren = "E0\nE1\nE2\nE3\nE4\nE5\nE6"
#strcn = "C0\nC1\nC2\nC3\nC4\nC5\nC6"#\nC7\nC8\nC9"
# print(str)
# print(re.split(r'([.?!])', str))

# str = 'project 3 abc'
# pw = 'projeact'
# print(str.find(pw))

# import random
# wdict = {}
# wdict['a'] = 'A'
# wdict['b'] = 'B'
# wdict['c'] = 'C'
# wdict['d'] = 'D'
# wdict['e'] = 'E'
# wdict['f'] = 'F'
# wdict['g'] = 'G'
# wdict['h'] = 'H'
# wdict['i'] = 'I'
# wdict['j'] = 'J'
# wdict['k'] = 'K'
# wdict['l'] = 'L'
# wdict['m'] = 'M'
# wdict['n'] = 'N'
# while len(wdict.keys())>1:
#     k1 = random.choice(list(wdict.keys()))
# #     del wdict[k1]
#     wdict.pop(k1)
#     print(list(wdict.keys()))