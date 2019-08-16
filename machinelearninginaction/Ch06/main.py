'''
@author: laiwei
'''
from numpy import *
import svmMLiA

'''
dataArr,labelArr = svmMLiA.loadDataSet('testSet.txt')
#b,alphas = svmMLiA.smoSimple(dataArr, labelArr, 0.6 , 0.001, 40)
b,alphas = svmMLiA.smoPNK(dataArr, labelArr, 0.6 , 0.001, 40)
#b,alphas = svmMLiA.smoP(dataArr, labelArr, 0.6 , 0.001, 40, kTup=('rbf', 1,5))
print(b)
print(alphas[alphas>0])
#for i in range(len(labelArr)):
#    if alphas[i]>0.0: print(dataArr[i],labelArr[i])
ws = svmMLiA.calcWs(alphas, dataArr, labelArr)
#print(min(mat(dataArr)[:,0]), max(mat(dataArr)[:,0]))
#print(min(mat(dataArr)[:,1]), max(mat(dataArr)[:,1]))


def xy(r,phi,x,y):
    return r*cos(phi)+x, r*sin(phi)+y
def yuan(ax,x,y,r,c='r'):
    phis=arange(0,6.28,0.01)
    ax.plot(*xy(r,phis,x,y), c, ls='-' )
def plot1():
    import matplotlib.pyplot as plt
    dataMat,labelMat=svmMLiA.loadDataSet('testSet.txt')
    dataArr = array(dataMat)
    n = shape(dataArr)[0] # num of samples
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,0]); ycord1.append(dataArr[i,1])
        else:
            xcord2.append(dataArr[i,0]); ycord2.append(dataArr[i,1])
    fig = plt.figure()
    ax = fig.add_subplot(111,aspect='equal')
    ax.scatter(xcord1, ycord1, s=20, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=20, c='green')
    
    #ax.add_artist(plt.Circle((4.65, 3.5), 0.3, color='b', fill=False))
    #ax.add_artist(plt.Circle(array(dataArr[5]), 0.3, color='b', fill=False))
    #yuan(ax, dataArr[5][0], dataArr[5][1], 0.3, 'g')
            
    for i in range(n):
        if alphas[i]>0.0:
            clr = 'b'
            if(labelMat[i]>0): clr='y'
            #yuan(ax, dataArr[i][0], dataArr[i][1], 0.3, clr)
            ax.add_artist(plt.Circle(array(dataArr[i]), 0.3, color=clr, fill=False))
    
    #x = arange(-3.0, 3.0, 0.1)
    #x = arange(min(mat(dataArr)[:,0]), max(mat(dataArr)[:,0]), 0.1)
    #y = (-b[0,0]-ws[0]*x)/ws[1]
    y = arange(min(mat(dataArr)[:,1]), max(mat(dataArr)[:,1]), 0.1)
    x = (-b[0,0]-ws[1]*y)/ws[0]
    ax.plot(x, y)
    
    plt.xlabel('x'); plt.ylabel('y');
    plt.title("%s samples" % n)
    plt.show()

plot1()
'''
'''
def plot2():
    import matplotlib.pyplot as plt
    dataMat,labelMat=svmMLiA.loadDataSet('testSetRBF.txt')
    dataArr = array(dataMat)
    n = shape(dataArr)[0] # num of samples
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,0]); ycord1.append(dataArr[i,1])
        else:
            xcord2.append(dataArr[i,0]); ycord2.append(dataArr[i,1])
    fig = plt.figure()
    ax = fig.add_subplot(111,aspect='equal')
    ax.scatter(xcord1, ycord1, s=20, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=20, c='green')
    
    for i in svInd:
        clr = 'b'
        if(labelMat[i]>0): clr='y'
        ax.add_artist(plt.Circle(array(dataArr[i]), 0.04, color=clr, fill=False))
    
    plt.xlabel('x'); plt.ylabel('y');
    plt.title("%s samples" % n)
    plt.show()
    
svInd = svmMLiA.testRbf()
plot2()
'''

svmMLiA.testDigits()
