#coding:utf-8
'''
Created on 2017年3月10日
@author: laiwei
'''
import pca
from numpy import *

def test1():
    dataMat = pca.loadDataSet('testSet.txt')
    #pca.plot1(dataMat)
    lowDMat, reconMat = pca.pca(dataMat, 1)
    pca.plot2(dataMat, reconMat)
    
#test1()

def test2():
    dataMat = pca.replaceNanWithMean()
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals
    covMat = cov(meanRemoved, rowvar=0)
    eigVals,eigVects = linalg.eig(mat(covMat))
    print(eigVals)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(eigVals[:8])
    plt.show()
    
test2()