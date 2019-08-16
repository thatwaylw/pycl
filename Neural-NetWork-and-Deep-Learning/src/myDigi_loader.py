# -*- coding: utf-8 -*-
'''
Created on 2017年3月15日
@author: laiwei
'''
from numpy import *
import operator
from os import listdir

def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def vectorized_results(j):
    e = zeros((10,1))
    e[j] = 1.0
    return e

def loadFolderTrain(folder='..\\data\\trainingDigits'):
    hwLabels = []
    trainingFileList = listdir(folder)           #load the training set
    m = len(trainingFileList)
    #trainingMat = zeros((m,1024))
    trainingArray = []
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        #trainingMat[i,:] = img2vector('%s/%s' % folder, fileNameStr)
        trainingArray.append(img2vector('%s/%s' % (folder, fileNameStr)))
    
    training_inputs = [reshape(x,(1024,1)) for x in trainingArray]    
    training_results = [vectorized_results(y) for y in hwLabels]
    training_data = list(zip(training_inputs,training_results))
        
    return training_data

def loadFolderTest(folder='..\\data\\testDigits'):
    hwLabels = []
    trainingFileList = listdir(folder)           #load the training set
    m = len(trainingFileList)
    #trainingMat = zeros((m,1024))
    trainingArray = []
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        #trainingMat[i,:] = img2vector('%s/%s' % folder, fileNameStr)
        trainingArray.append(img2vector('%s/%s' % (folder, fileNameStr)))
    
    training_inputs = [reshape(x,(1024,1)) for x in trainingArray]    
    training_data = list(zip(training_inputs,hwLabels))
        
    return training_data
        
