# -*- coding: UTF-8 -*-
import numpy as np

sentence = (
    'The morning had dawned clear and cold with a crispness that hinted at the end of summer They '
#     'set forth at daybreak to see a man beheaded twenty in all and Bran rode among them nervous with '
#     'excitement This was the first time he had been deemed old enough to go with his lord father and his '
#     'brothers to see the king’s justice done It was the ninth year of summer and the seventh of Bran’s life'
) # 就是1句，1个str

word_dict = {w: i for i, w in enumerate(list(set(sentence.split())))}
number_dict = {i: w for i, w in enumerate(list(set(sentence.split())))}
n_class = len(word_dict)
n_step = 5 #len(sentence.split())
n_hidden = 5

def make_batch(sentence):
    input_batch = []
    target_batch = []

    words = sentence.split()
    for i, word in enumerate(words[:-1]):
        input = [word_dict[n] for n in words[:(i + 1)]]
        input = input + [0] * (n_step - len(input))     # !!! n_step<len(sen) 时，input长度不一致了！！！
        target = word_dict[words[i + 1]]
        input_batch.append(np.eye(n_class)[input])
        target_batch.append(np.eye(n_class)[target])

    return input_batch, target_batch

input_batch, target_batch = make_batch(sentence)

print(sentence)
print(len(input_batch), input_batch[0].shape)
print(len(target_batch), target_batch[0].shape)