# -*- coding: utf-8 -*-
'''
Created on 2017年3月14日
@author: laiwei
'''
import numpy
import random

sizes = [2,3,1]
num_layers = len(sizes)
biases = [numpy.random.randn(y, 1) for y in sizes[1:]]
weights = [numpy.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

print(list(zip(sizes[:-1], sizes[1:])))
print(biases)
print(weights)