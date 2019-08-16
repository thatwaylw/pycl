#coding:utf-8
'''
@author: laiwei
date: 2017年3月4日
'''

import adaboost
from numpy import *

#datMat, classLabels = adaboost.loadSimpData()
#adaboost.plotData(datMat, classLabels)
datMat, classLabels = adaboost.loadDataSet('horseColicTraining2.txt')

#D = mat(ones((5, 1))/5)
#bestStump,minError,bestClasEst = adaboost.buildStump(datMat, classLabels, D)
#print(bestStump);print(minError);print(bestClasEst)

weakClassArr,aggClassEst = adaboost.adaBoostTrainDS(datMat, classLabels,37)
#aggClassEst[0,0] = -0.2
#classLabels[0] = -1
#print(weakClassArr);print(aggClassEst)
#print(adaboost.adaClassify([[0,0],[5,5]], weakClassArr))

# 当预测label按大小排序，对应真实label不是先全部-1，再全部+1，而是中间有错乱时，曲线下弯
#adaboost.plotROC(mat(classLabels), classLabels)
adaboost.plotROC(aggClassEst.T, classLabels) 


testdatMat, testclassLabels = adaboost.loadDataSet('horseColicTest2.txt')
testResult = adaboost.adaClassify(testdatMat, weakClassArr)
errArr=mat(ones((len(testclassLabels),1)))
print(errArr[testResult != mat(testclassLabels).T].sum(), "of total", shape(testclassLabels))
