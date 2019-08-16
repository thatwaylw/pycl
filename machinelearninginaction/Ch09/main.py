#coding:utf-8
'''
Created on 2017年3月7日
@author: laiwei
'''
from numpy import *
import regTrees

def test1():
    testMat=mat(eye(4))
    print(testMat)
    
    print(testMat[:,2])
    print(testMat[:,2]<0.5) # 第二列小于0.5的情况，有True/False组成，结果m行1列的矩阵
    print(nonzero(testMat[:,2]<0.5))
    print(nonzero(testMat[:,2]<0.5)[0]) #nonzero(mat)，如果有k个非零值，返回(a1=[x1,x2,...xk], a2=(y1,2y,...yk))，xi,yi为非零值对应的行idx和列idx
    print(testMat[nonzero(testMat[:,2]<0.5)[0],:]) #可能为空 []，比如改成<-0.5
    print(testMat[nonzero(testMat[:,2]<0.5)[0],:][0]) #如果上面为空的话，本句出错，下标越界
    #dataSet[nonzero(dataSet[:,feature] > value)[0],:][0]
    print('--')
    mat0,mat1=regTrees.binSplitDataSet(testMat,1,0.5)
    print(mat0) #第一个满足第1维特征>0.5的行--->需要修改？？？
    print(mat1) #第一个满足第1维特征<=0.5的行
    
#test1()

def test10():
    myDat = regTrees.loadDataSet('ex00.txt')  
    fy = mat(myDat)[0:5,1]
    print(fy)
    afy = fy[0]
    print(afy)
    fy[1]=fy[0]
    print('fy', fy)
    print('fy.T', fy.T)
    print('fy.tolist()', fy.tolist())
    print('fy.T.tolist()', fy.T.tolist())
    print('fy.T.tolist()[0]', fy.T.tolist()[0])
    #print(set(fy)) #出错：unhashable type: 'matrix'
    #print(set(fy.T.tolist()))#出错：TypeError: unhashable type: 'list'
    print(set(fy.T.tolist()[0]))  #这样可以！！实现了对fy里元素的去重复！
    my = map(float, fy);  print(set(my))  #这样也可以！！实现了对fy里元素的去重复！
    
#test10()

def test2():
    ''' line = '0.530897\t0.893462'
    curLine = line.strip().split('\t')
    print(curLine)
    fltLine = map(float, curLine) #在python3里面变成map了。。。错了
    print(list(curLine))
    print(set(fltLine)) '''
    
    myDat = regTrees.loadDataSet('exp.txt')    #200*2 float, exp和ex2也差不多，y的方差更小
    #print(myDat[2])    #print(list(myDat[2]))一样的
    print(shape(myDat))
    #print(m1[:,-1])    #print(m1[5,:]) # 用mat()转成矩阵才能这样用
    myMat = mat(myDat)
    #retTree = regTrees.createTree(myMat, ops=(1000,10)) #(0,1)就是每个点都分了一个叉，典型的overfitting
    #retTree = regTrees.createTree(myMat, ops=(0.2,4)) #ex2比ex00分布差不多，y的取值大了100倍，因此用10000,4和原来的效果差不多
    #print(retTree)
    
    retTree = regTrees.createTree(myMat, ops=(10,4))
    testDat = mat(regTrees.loadDataSet('ex2test.txt'))  #ex2test.txt的数据分布范围和ex2很接近，真实的测试集
    pruned_Tree = regTrees.prune(retTree, testDat)
    print(pruned_Tree)
    
    #regTrees.plot1(myMat)
    regTrees.plot1withTree(myMat, retTree)
    regTrees.plot1withTree(myMat, pruned_Tree)
    
#test2()

def test3():
    myDat = mat(regTrees.loadDataSet('exp2.txt'))    #200*2 float
    print(shape(myDat))
    retTree = regTrees.createTree(myDat, regTrees.modelLeaf, regTrees.modelErr,(1,10))
    print(retTree)
    #regTrees.plot1(myDat)
    regTrees.plot1withTree_Linear(myDat, retTree)

#test3()

def test4():
    trainMat=mat(regTrees.loadDataSet('bikeSpeedVsIq_train.txt'))
    testMat=mat(regTrees.loadDataSet('bikeSpeedVsIq_test.txt'))
    #regTrees.plot1(testMat)
    myTree=regTrees.createTree(trainMat, ops=(1,20))
    yHat = regTrees.createForeCast(myTree, testMat[:,0])
    print(corrcoef(yHat, testMat[:,1],rowvar=0)[0,1])
    #regTrees.plot1withTree(trainMat, myTree)
    
    myTree=regTrees.createTree(trainMat, regTrees.modelLeaf,regTrees.modelErr,(1,20))
    yHat = regTrees.createForeCast(myTree, testMat[:,0],regTrees.modelTreeEval)
    print(corrcoef(yHat, testMat[:,1],rowvar=0)[0,1])
    print(myTree)
    regTrees.plot1withTree_Linear(trainMat, myTree)
    
    ws,X,Y=regTrees.linearSolve(trainMat)
    print(ws)
    for i in range(shape(testMat)[0]):
        yHat[i]=testMat[i,0]*ws[1,0]+ws[0,0]
    print(corrcoef(yHat, testMat[:,1],rowvar=0)[0,1])    
    
#test4()

def test5():
    myDat = mat(regTrees.loadDataSet('sine.txt'))
    regTrees.plot1(myDat)
    
test5()