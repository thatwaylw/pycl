#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt

# Show Attention
#sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']
sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]

attention = [
#     [7.8082230e-05, 1.2828252e-15, 3.5475309e-03, 9.9637443e-01, 3.6211913e-16],
#     [1.0334127e-24, 2.2117861e-30, 1.0754508e-20, 0.0000000e+00, 1.0000000e+00],
#     [1.1209729e-12, 1.9054230e-25, 9.9990332e-01, 9.6653835e-05, 1.9925226e-24],
#     [1.0000000e+00, 1.7071990e-09, 4.0836066e-26, 5.6329858e-26, 1.3447151e-27],
#     [2.8208853e-28, 2.4336617e-02, 9.4116023e-03, 9.6625173e-01, 3.4710455e-28]]
    [0.18534335, 0.7909841,  0.02367259],
    [0.66865426, 0.2956973,  0.03564842],
    [0.6821927,  0.27037182, 0.04743558],
    [0.22904281, 0.3320987,  0.43885848],
    [0.16660911, 0.34222147, 0.49116942],
    [0.18240128, 0.3107807,  0.506818  ]]
 
# def showatt():
#     fig = plt.figure(figsize=(5, 5))
#     ax = fig.add_subplot(1, 1, 1)
#     ax.matshow(attention, cmap='viridis')
#     ax.set_xticklabels([''] + sentences[0].split(), fontdict={'fontsize': 14})
#     ax.set_yticklabels([''] + sentences[2].split(), fontdict={'fontsize': 14})
#     plt.show()
def showatt():
    fig = plt.figure(figsize=(6, 3)) # [batch_size, n_step]
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')
    ax.set_xticklabels([''] + ['first_word', 'second_word', 'third_word'], fontdict={'fontsize': 14}, rotation=90)
    ax.set_yticklabels([''] + ['batch_1', 'batch_2', 'batch_3', 'batch_4', 'batch_5', 'batch_6'], fontdict={'fontsize': 14})
    plt.show()    
    
showatt()

'''
pltdat = [['cat', -0.7235305, -0.6633502],
    ['fish', -2.44461, 0.18892916],
    ['movie', 0.10255699, 0.19957556],
    ['animal', -0.05785167, -1.9238065],
    ['music', 2.0552208, 0.83697027],
    ['dog', 0.8142485, -1.165833],
    ['eyes', 1.2442197, -3.010993],
    ['love', -1.524863, -0.47248042],
    ['milk', 1.3778304, -0.6343653],
    ['i', 0.87626874, -0.9919268],
    ['like', -1.0577252, -0.9000024],
    ['hate', -0.8326449, -1.940324],
    ['book', -0.35238674, 3.3317263],
    ['apple', 0.06284091, -1.2931379]]
'''
'''
pltdat = [
['like', -1.233120, -0.840287],
['cat', 0.018793, 1.664447],
['i', 1.494483, -0.380628],
['very', 2.466545, -2.292530],
['a', 0.419084, -1.404652],
['hate', -1.172227, -0.745308],
['much', -1.240941, 0.223601],
['lot', -1.217374, 2.215158],
['dog', 0.079165, 1.444849],
['he', 1.434572, -0.213770],
['love', -1.120936, -0.714515],
['apple', -0.176007, 2.436146],
['she', 1.318725, -0.255223],
['desperately', -0.695489, -0.374114]]
          
for pd in pltdat:
    label = pd[0]
    x = pd[1]
    y = pd[2]
    print(label,'\t', x,'\t', y)
    plt.scatter(x, y)
    plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
plt.show()
'''