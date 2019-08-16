#coding:utf-8
'''
Created on Mar 8, 2011

@author: Peter
'''
from numpy import *
from numpy import linalg as la

def loadExData():
    return[[0, 0, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1],
           [1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 1, 0, 0]]
    
def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]
    
def ecludSim(inA,inB):
    return 1.0/(1.0 + la.norm(inA - inB))

def pearsSim(inA,inB):
    if len(inA) < 3 : return 1.0
    return 0.5+0.5*corrcoef(inA, inB, rowvar = 0)[0][1]

def cosSim(inA,inB):
    num = float(inA.T*inB)
    denom = la.norm(inA)*la.norm(inB)
    return 0.5+0.5*(num/denom)

def standEst(dataMat, user, simMeas, item): #dataMat[user, item]=0，需要预估一个分数，作为推荐的依据
    n = shape(dataMat)[1] #item的个数
    simTotal = 0.0; ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0: continue  #如果user没有打分过j，跳过
        overLap = nonzero(logical_and(dataMat[:,item].A>0, \
                                      dataMat[:,j].A>0))[0] #所有当前item和j都打过分的用户idx们
        if len(overLap) == 0: similarity = 0
        else: similarity = simMeas(dataMat[overLap,item], \
                                   dataMat[overLap,j])  #给定user，item，item和j的相似度
        print('the %d and %d similarity is: %f -- %f' % (item, j, similarity, userRating))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal
    
def svdEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    U,Sigma,VT = la.svd(dataMat) # Sigma仅仅是个array
    Sig4 = mat(eye(4)*Sigma[:4]) #arrange Sig4 into a diagonal matrix，取前4个值，变成4x4对角阵
    xformedItems = dataMat.T * U[:,:4] * Sig4.I  #(n*m) * (m*4) * (4*4) = (n*4), item数没变，user数压缩了
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0 or j==item: continue
        similarity = simMeas(xformedItems[item,:].T,\
                             xformedItems[j,:].T)
        print('the %d and %d similarity is: %f -- %f' % (item, j, similarity, userRating))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal

def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    unratedItems = nonzero(dataMat[user,:].A==0)[1]#find unrated items, nonzero返回两个array，分别是矩阵中非零值的行idx们，列idx们，此处取列，对应不同items
    if len(unratedItems) == 0: return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N] #lambda其实是一个简单的函数

def printMat(inMat, thresh=0.8):
    for i in range(32):
        linstr = '';
        for k in range(32):
            if float(inMat[i,k]) > thresh:
                linstr += '1'
            else: linstr += '0'
        print(linstr)

def imgCompress(numSV=3, thresh=0.8):
    myl = []
    for line in open('0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = mat(myl)
    print("****original matrix******")
    printMat(myMat, thresh)
    U,Sigma,VT = la.svd(myMat)
    SigRecon = mat(zeros((numSV, numSV)))
    for k in range(numSV):#construct diagonal matrix from vector
        SigRecon[k,k] = Sigma[k]
    reconMat = U[:,:numSV]*SigRecon*VT[:numSV,:]
    print("****reconstructed matrix using %d singular values******" % numSV)
    printMat(reconMat, thresh)
    
def test1():
    U,Sigma,VT=linalg.svd([[1, 1],[7, 7]])
    print(U); print(Sigma); print(VT)
    
    data = loadExData()
    U,Sigma,VT=linalg.svd(data)
    print(Sigma)
    
    Sig3=mat([[Sigma[0], 0, 0],[0, Sigma[1], 0], [0, 0, Sigma[2]]])
    data1 = U[:,:3]*Sig3*VT[:3,:]
    print(data1)
    
def test2():
    myMat=mat(loadExData())
    print(ecludSim(myMat[:,0],myMat[:,4]))
    print(ecludSim(myMat[:,0],myMat[:,0]))
    print(cosSim(myMat[:,0],myMat[:,4]))
    print(cosSim(myMat[:,0],myMat[:,0]))
    print(pearsSim(myMat[:,0],myMat[:,4]))
    print(pearsSim(myMat[:,0],myMat[:,0]))
    
def test3():
    myMat=mat(loadExData())
    myMat[0,1]=myMat[0,0]=myMat[1,0]=myMat[2,0]=4;myMat[3,3]=2
    print(myMat)
    res = recommend(myMat, 2)
    print(res)
 
def test4():
    U,Sigma,VT=la.svd(mat(loadExData2()))
    print(Sigma)
    Sig2=Sigma**2
    print(sum(Sig2))
    print(sum(Sig2[:3]))
    
    myMat=mat(loadExData2())
    res = recommend(myMat, 1, estMethod=svdEst)
    print(res)
    res = recommend(myMat, 1, estMethod=svdEst,simMeas=pearsSim)
    print(res)
 
def test5():
    imgCompress(2)
    
#test1()
#test2()
#test3()
#test4()
test5()