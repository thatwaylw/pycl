#coding:utf-8

import io
with io.open('./data/tencent_chat_a.txt', 'r', encoding='utf-8') as f:
    source_data = f.read()

line_list = source_data.split('\n')
for i in range(15):
    line = line_list[i]
    buf = ''
    for w in line.split():		#按空格分隔，原空格没有了，多个空格算一个
        buf += w
        buf += '_'
    print(buf)
    
    buf1 = ''
    for w in line:				#每个字符分隔，所有空格原样保留
        buf1 += w
    print(buf1)
    
print('\n测试中文相关')
while True:
    line = input('我：')#line = raw_input('我：') //python2 only
    if line == '': break
    print('Robot: '+line)
    print(line.split())
    buf=''
    for w in line.split():
        buf += w
    print(buf)