# -*- coding: utf-8 -*-
"""
mnist_loader
-----------------------------

A library to load the MNIST image data. For detail of the data structures that ara returned,
see the doc string for "load_data" and "load_data_wrapper". In practice "load_data_wrapper"
is the function usually by our neural network code
"""

import pickle
import gzip
import numpy

def load_data(dataName):
    f = gzip.open(dataName, 'rb')
    training_data,validation_data,test_data=pickle.load(f,encoding="iso-8859-1")
    f.close()
    return(training_data,validation_data,test_data)

def load_data_wrapper(dataName):
    tr_d,va_d,te_d = load_data(dataName)
    training_inputs = [numpy.reshape(x,(784,1)) for x in tr_d[0]]
    training_results = [vectorized_results(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs,training_results))
    validation_inputs = [numpy.reshape(x,(784,1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs,va_d[1]))
    test_inputs = [numpy.reshape(x,(784,1)) for x in te_d[0]]
    test_data = list(zip(test_inputs,te_d[1]))
    print('%d train, %d test, %d validation' % (len(training_inputs), len(test_inputs), len(validation_inputs)))
    return(training_data,validation_data,test_data)

def vectorized_results(j):
    e = numpy.zeros((10,1))
    e[j] = 1.0
    return e
# try:
#     training_data, validation_data, test_data = load_data_wrapper()
# except Exception as e:
#     pass
