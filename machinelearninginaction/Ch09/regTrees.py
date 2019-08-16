#coding:utf-8
'''
Created on Feb 4, 2011
Tree-Based Regression Methods
@author: Peter Harrington
'''
from numpy import *

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        #fltLine = map(float,curLine) #map all elements to float()
        #dataMat.append(fltLine)
        lineArr =[]
        for i in range(len(curLine)):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
    return dataMat

def plot1(dataMat):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:,-2], dataMat[:,-1], s=15)
    plt.show()

def estimateByRegTree(dat, regTree):
    if not isTree(regTree): return regTree
    if dat[0, regTree['spInd']] > regTree['spVal']:  #dat=[[x,y]], dat[0,0]=x, dat[0,1]=y
        return estimateByRegTree(dat, regTree['left'])
    else:
        return estimateByRegTree(dat, regTree['right'])
    
def estimateByRegTree_Linear(dat, regTree):
    if not isTree(regTree):
        m,n = shape(dat) #dat=[[x,y]]
        X = mat(ones((1,n)))  #X=[[1,1]]
        X[:,1:n] = dat[:,0:n-1]; #X=[[1,x]]
        return X*regTree
    if dat[0, regTree['spInd']] > regTree['spVal']:  #dat=[[x,y]], dat[0,0]=x, dat[0,1]=y
        return estimateByRegTree_Linear(dat, regTree['left'])
    else:
        return estimateByRegTree_Linear(dat, regTree['right'])
    
def plot1withTree(dataMat,regTree):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:,-2], dataMat[:,-1], s=15)
    
    m,n=shape(dataMat)
    estMat = zeros(m)
    for i in range(m):
        estMat[i]=estimateByRegTree(dataMat[i,:], regTree)
    ax.scatter(dataMat[:,-2], estMat, c='r', marker='.', s=1)
    
    plt.show()
    
def plot1withTree_Linear(dataMat,regTree):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:,-2], dataMat[:,-1], s=15)
    
    m,n=shape(dataMat)
    estMat = zeros(m)
    for i in range(m):
        estMat[i]=estimateByRegTree_Linear(dataMat[i,:], regTree)
    ax.scatter(dataMat[:,-2], estMat, c='r', marker='.', s=1)
    
    plt.show()            

def binSplitDataSet(dataSet, feature, value):
    #mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:][0]#第一个满足第feature维特征>value的行--->需要修改？？？
    #mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:][0]#第一个满足第feature维特征<=value的行--->需要修改？？？
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:]#所有满足第feature维特征>value的行
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:]#所有满足第feature维特征<=value的行
    return mat0,mat1

def regLeaf(dataSet):#returns the value used for each leaf
    return mean(dataSet[:,-1])                    #最后一列（y值）的均值

def regErr(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0] #最后一列（y值）的均方差*样本个数=平方误差

def linearSolve(dataSet):   #helper function used in two places
    m,n = shape(dataSet)
    X = mat(ones((m,n))); Y = mat(ones((m,1)))#create a copy of data with 1 in 0th postion
    X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:,-1]#and strip out Y
    xTx = X.T*X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws,X,Y

def modelLeaf(dataSet):#create linear model and return coeficients
    ws,X,Y = linearSolve(dataSet)
    return ws                           #线性模型的系数 y=w0+w1*x

def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat,2))       #线性模型的平方误差 sum((y-yhat)**2)

def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    tolS = ops[0]; tolN = ops[1]
    #if all the target variables are the same value: quit and return value
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1: #dataSet最后一维（y值）的所有取值可能为唯一一种的时候，exit cond 1
        return None, leafType(dataSet)  #直接返回最佳idx=空，最佳val=dataSet的y的均值
    m,n = shape(dataSet) #m=200个样本，n=2维，(x,y)
    #the choice of the best feature is driven by Reduction in RSS error from mean
    S = errType(dataSet) #分割前，dataSet的y的方差
    bestS = inf; bestIndex = 0; bestValue = 0
    for featIndex in range(n-1): #对除了最后一维y的所有维度，当前只有0 （x）
        for splitVal in set(map(float, dataSet[:,featIndex])): #splitVal为x的所有可能取值做个遍历
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal) #满足x>splitVal的所有行，和另一半
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): continue #如果两半，任一个都少于tolN=4行，继续下一个可能值
            newS = errType(mat0) + errType(mat1) #否则，两半都不会特别小，计算两半的总方差
            if newS < bestS: 
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    #if the decrease (S-bestS) is less than a threshold don't do the split
    if (S - bestS) < tolS: # 如果分割之后两半的y方差之和,比分割前y方差，没有显著的降低，直接返回最佳idx=空，最佳val=dataSet的y的均值
        return None, leafType(dataSet) #exit cond 2
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue) #按照刚才的最佳维度，最佳门限进行分割，将dataSet分为两半，
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  #再次确保两半都足够大，exit cond 3
        return None, leafType(dataSet) #两半有一个太小，直接返回最佳idx=空，最佳val=dataSet的y的均值
    return bestIndex,bestValue#返回分割的最佳维度（这里肯定是0，x）和最佳门限。returns the best feature to split on and the value used for that split

def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):#assume dataSet is NumPy Mat so we can array filtering
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)#choose the best split
    if feat == None: return val #if the splitting hit a stop condition return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val) #递归，左右子树分别分割后的两半，再去各自递归建立树
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree  

def isTree(obj):
    return (type(obj).__name__=='dict')

def getMean(tree):
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0
    
def prune(tree, testData):
    if shape(testData)[0] == 0: return getMean(tree) #if we have no test data collapse the tree
    if (isTree(tree['right']) or isTree(tree['left'])):#if the branches are not trees try to prune them
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): tree['right'] =  prune(tree['right'], rSet)
    #if they are now both leafs, see if we can merge them
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) +\
            sum(power(rSet[:,-1] - tree['right'],2))
        treeMean = (tree['left']+tree['right'])/2.0
        errorMerge = sum(power(testData[:,-1] - treeMean,2))
        if errorMerge < errorNoMerge: 
            print("merging")
            return treeMean
        else: return tree
    else: return tree
    
def regTreeEval(model, inDat):
    return float(model)

def modelTreeEval(model, inDat):#inDat=testData[i]，输入时已经去除了最后一列y
    n = shape(inDat)[1]
    X = mat(ones((1,n+1)))
    X[:,1:n+1]=inDat
    return float(X*model)

def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree): return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']): return treeForeCast(tree['left'], inData, modelEval)
        else: return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']): return treeForeCast(tree['right'], inData, modelEval)
        else: return modelEval(tree['right'], inData)
        
def createForeCast(tree, testData, modelEval=regTreeEval):  #testData输入时已经去除了最后一列y
    m=len(testData)
    yHat = mat(zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat