'''
Created on Jun 1, 2011

@author: Peter Harrington
'''
from numpy import *

def plot1(dataMat):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:,-2], dataMat[:,-1], s=15)
    plt.show()
    
def plot2(dataMat, reconMat):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:,0], dataMat[:,1], s=25)
    ax.scatter(reconMat[:,0], reconMat[:,1], s=5, c='r')
    plt.show()    
    
def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    #datArr = [map(float,line) for line in stringArr]
    datArr = [list(map(float,line)) for line in stringArr]
    return mat(datArr)

def pca(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat, axis=0) # [E(x),E(Y)] ==> 1*2矩阵
    meanRemoved = dataMat - meanVals #remove mean ==> m*2矩阵 - 1*2 矩阵，还是m*2矩阵
    covMat = cov(meanRemoved, rowvar=0)  # 协方差 E[(X-E(X))(Y-E(Y))] 是只能两维吗？ 得到一个2x2方阵，对角阵
    eigVals,eigVects = linalg.eig(mat(covMat)) #1*2，2*2
    eigValInd = argsort(eigVals)            #sort, sort goes smallest to largest
    eigValInd = eigValInd[:-(topNfeat+1):-1]  #cut off unwanted dimensions; topNfeat=1 ===> x[-1:-2:-1] 取了最后一个
    redEigVects = eigVects[:,eigValInd]       #reorganize eig vects largest to smallest，注意，取到若干个列向量（2*1）
    lowDDataMat = meanRemoved * redEigVects #映射到新维度空间 并降维 (m*2) * (2*1) = (m*1)
    reconMat = (lowDDataMat * redEigVects.T) + meanVals #又映射会原来的维度空间（为了可视化显示） (m*1) * (1*2) = (m*2) 并加回均值以显示对齐
    return lowDDataMat, reconMat

def replaceNanWithMean(): 
    datMat = loadDataSet('secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i]) #values that are not NaN (a number)
        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal  #set NaN values to mean
    return datMat
