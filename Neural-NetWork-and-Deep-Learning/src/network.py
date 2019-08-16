# -*- coding: utf-8 -*-
"""
network.py
-----------------------------------------------

A module to implement the stochastic gradient descent learning algorithm for feedforword neural network.
Gradients are calcutated using backpropagation. Note that I have focused on making the code simple,
easily readable,and easily modifiable. It is not optimized,and omit many desirable deatures
"""

import numpy
import random

class Network(object):

    def __init__(self, sizes):
        """
        The list "sizes" contains the number of neurons in the respective layers of the network.
        For example,if the list was [2,3,1] then it would be a three-layer network,with the first
        layer containing 2 neurons,the second layer 3 neurons,and the third layer 1 neuron. The biases
        and the weights for the network are initialized randomly,using a Gaussian distribution with mean 0,
        and variance 1. Note that the first won't set any biases for those neurons,since biases are only
        ever used in computing the outputs from later layers.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [numpy.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [numpy.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        Train the neural network using mini-bath stochastic gradient descent.
        parameters:
            training_data:a list of tuples "(x,y)" representing the training input and the desired output
            test_data:if it is provided then the network will be evaluated against the test data after
            each epoch ,and partial proess printed out.
        """
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}:{1}/{2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """
        Update the network's weights and biases by applying gradient descent using backpropagation
        to a single mini batch.abs
        parpmeters:
            mini_batch: a list of tuples "(x,y)"对应书中所写的y(x)，不是数值上的x,y
            eta: the learning rate
        """
        nabla_b = [numpy.zeros(b.shape) for b in self.biases]
        nabla_w = [numpy.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b,delta_nabla_w = self.backprop(x,y)
            nabla_b = [nb+dnb for nb,dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw,dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self,x,y):
        """
        Return a tuple "(nable_b,nable_w)" representing the gradient for the cost function C_x.
        "nable_b" and "nable_w" are layer by layer lists of numpy arrays,similar to self.biases 
        and self.weights
        """
        nabla_b = [numpy.zeros(b.shape) for b in self.biases]
        nabla_w = [numpy.zeros(w.shape) for w in self.weights]
        #feedforward
        activation = x
        activations = [x]  #list to store all the activations,layer by layer
        zs = [] #list to store all the z vectors,layer by layer
        for b, w in zip(self.biases, self.weights):
            z = numpy.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        #backward pass
        delta = self.cost_derivative(activations[-1],y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = numpy.dot(delta, activations[-2].transpose())
        #Note that the variable l in the loop below is used a little differently to the notation in
        #Chapter 2 of the book.Here,second-last layer and so on.It's a renumbering of the scheme in
        #the book,used here to take advantage of the fact that Python can use negative indices in lists
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = numpy.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = numpy.dot(delta, activations[-l-1].transpose())
        return (nabla_b,nabla_w)

    def cost_derivative(self, output_activations, y):
        """
        Return the vector of partial dervatives \partial C_x /partial a for the output activations
        """
        return (output_activations-y)

    def evaluate(self,test_data):
        """
        Return the number of test inputs for which the neural network outputs the correct result.
        Note that the neural network's output is assumed to be the index of whichever neuron
        in the final layer has the highest activation
        """
        test_result = [(numpy.argmax(self.feedforword(x)) ,y) for (x, y) in test_data]
        return sum(int(x ==y) for (x, y) in test_result)

    def feedforword(self, a):
        """return the output of network if a is input"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(numpy.dot(w, a) + b)
        return a


def sigmoid(z):
    """
    The sigmoid function
    """
    return 1.0/(1.0 + numpy.exp(-z))

def sigmoid_prime(z):
    """
    Derivative of the sigmoid function
    """
    return sigmoid(z)*(1-sigmoid(z))


import mnist_loader
if __name__ == "__main__":
    dataName = "..\\origsrcdat\\data\\mnist.pkl.gz" #50000 train, 10000 test, 10000 validation
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper(dataName)
    net = Network([784, 30, 10])
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)