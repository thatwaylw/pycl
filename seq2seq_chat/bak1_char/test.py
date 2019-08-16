#coding:utf-8
'''中文注释'''
'''
import tensorflow as tf
max_value = tf.reduce_max([1, 3, 3, 2])
with tf.Session() as sess:
    max_value = sess.run(max_value)
    print(max_value)
'''
import io
with io.open('./data/letters_source.txt', 'r', encoding='utf-8') as f:
    source_data = f.read()
with io.open('./data/letters_target.txt', 'r', encoding='utf-8') as f:
    target_data = f.read()

def extract_character_vocab(data):
    '''
    构造映射表
    '''
    special_words = ['<PAD>', '<UNK>', '<GO>',  '<EOS>']

    set_words = list(set([character for line in data.split('\n') for character in line]))
    # 这里要把四个特殊字符添加进词典
    int_to_vocab = {idx: word for idx, word in enumerate(special_words + set_words)}
    vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}

    return int_to_vocab, vocab_to_int


# 构造映射表
source_int_to_letter, source_letter_to_int = extract_character_vocab(source_data)
target_int_to_letter, target_letter_to_int = extract_character_vocab(target_data)

print(source_int_to_letter)
#print(source_letter_to_int) 一样的
#print(target_int_to_letter)	一样的
print(target_letter_to_int)

# 对字母进行转换
source_int = [[source_letter_to_int.get(letter, source_letter_to_int['<UNK>'])
               for letter in line] for line in source_data.split('\n')]
#print(source_int[:5])

target_int = [[target_letter_to_int.get(letter, target_letter_to_int['<UNK>'])
               for letter in line] + [target_letter_to_int['<EOS>']] for line in target_data.split('\n')]
print(target_int[:5])

print(len(source_int), len(target_int))

print('根据两点坐标计算直线斜率k，截距b：')
while True:
    line = raw_input()
    if line == '': break
    print('Robot: '+line)
    x1, y1, x2, y2 = (float(x) for x in line.split())
    k = (y2 - y1) / (x2 - x1)
    b = y1 - k * x1
    print('斜率:{}，截距:{}'.format(k, b))
