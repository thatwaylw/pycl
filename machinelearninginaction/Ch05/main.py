'''
@author: laiwei
'''
import logRegres
from numpy import *

dataArr,labelMat=logRegres.loadDataSet()
#print(dataArr)
#print(labelMat)

#weights = logRegres.gradAscent(dataArr,labelMat)
#print(weights)
#logRegres.plotBestFit(weights.getA())

weights = logRegres.stocGradAscent1(array(dataArr),labelMat)
print(weights)
logRegres.plotBestFit(weights)

#logRegres.multiTest();