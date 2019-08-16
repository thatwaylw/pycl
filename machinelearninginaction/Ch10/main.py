#coding:utf-8
'''
Created on 2017年3月7日
@author: laiwei
'''
import kMeans
from numpy import *

def test1():
    datMat = mat(kMeans.loadDataSet('testSet.txt'))
    print(min(datMat[:,0]))
    print(kMeans.randCent(datMat, 2))
    print(kMeans.distEclud(datMat[0], datMat[1]))
    
    #myCentroids, clustAssing = kMeans.kMeans(datMat,3)
    myCentroids, clustAssing = kMeans.biKmeans(datMat,4)
    print(myCentroids)
    
    kMeans.plot1(datMat, myCentroids)

#test1() 

def test2():
    #kMeans.geoGrab('1 VA Center', 'Augusta, ME1')
    #kMeans.testURLLib()
    kMeans.clusterClubs(5)
    
test2()