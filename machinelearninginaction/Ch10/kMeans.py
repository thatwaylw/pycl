#coding:utf-8
'''
Created on Feb 16, 2011
k Means Clustering for Ch10 of Machine Learning in Action
@author: Peter Harrington
'''
from numpy import *

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        #fltLine = map(float,curLine) #map all elements to float() # map()在python3里不一样！！！。
        #dataMat.append(fltLine)
        lineArr =[]
        for i in range(len(curLine)):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
    return dataMat

def plot1(dataMat, centroids):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:,0], dataMat[:,1], s=15)
    ax.scatter(centroids[:,0], centroids[:,1], s=45, c='r', marker='x')
    plt.show()

def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2))) #la.norm(vecA-vecB)

def randCent(dataSet, k):
    n = shape(dataSet)[1]   #样本维度2
    centroids = mat(zeros((k,n)))#create centroid mat
    for j in range(n):#create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:,j])  # 第j维最小值，标量
        rangeJ = float(max(dataSet[:,j]) - minJ)  #标量
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1)) # k行1列[0,1)之间的随机数
    return centroids
    
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0] #80个点
    clusterAssment = mat(zeros((m,2)))#80行，每行存：最近的中心编号j，以及最近的距离平方
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):#for each data point assign it to the closest centroid
            minDist = inf; minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])  #第j个中心和第i个样本点之间的距离
                if distJI < minDist:
                    minDist = distJI; minIndex = j  # 对每个点i，找最近的中心j
            if clusterAssment[i,0] != minIndex: clusterChanged = True   #如果此轮迭代中，点i的最近中心编号j变了。。
            clusterAssment[i,:] = minIndex,minDist**2  #更新此轮迭代中的，点i的两个数据：最近中心编号j和相应的距离平方
        #print(centroids)
        for cent in range(k):#recalculate centroids
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]#dataSet的子集，最近中心编号j==cent
            centroids[cent,:] = mean(ptsInClust, axis=0) #18行2列，求平均成1行2列，不加axis是全部元素求均值变成一个标量，=0按列求均值，把行求掉，=1按行求均值
    return centroids, clusterAssment

def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0] #80个点
    clusterAssment = mat(zeros((m,2)))#80行，每行存：最近的中心编号j，以及最近的距离平方，j一开始全部是0（质心=初始中心，在centList里编号为0）
    centroid0 = mean(dataSet, axis=0).tolist()[0] #所有点的平均中心（质心），作为第一初始中心
    centList =[centroid0] #create a list with one centroid （质心=初始中心，在centList里编号为0）
    for j in range(m):#calc initial Error
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2 #先算好每个点j到初始中心c0的距离平方
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)): # 对当前每个中心
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]#dataSet的子集，最近中心编号=i的所有点
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)#对当前每个簇做一次2-kMean
            sseSplit = sum(splitClustAss[:,1])#当前最佳2分类的误差平方和
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1]) #（未分部分的误差）当前中心i之外所有簇的总误差平方和
            print("sseSplit, and notSplit: ",sseSplit,sseNotSplit) #两个加起来就是当前把i一分为二之后的总误差
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat  #原来的i分成两个，这里包含两个新中心的的坐标mat（2x2）
                bestClustAss = splitClustAss.copy() # 样本点个数仅仅是原来i簇的样本点个数，第一维最近中心编号，只有0,1两种取值（新2中心的本轮内部编号）
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #要先看下面cenList更新的两行，1成了追加的那个，编号变成了最后
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit#0成了原来替代掉的，编号就是bestCentToSplit
        print('the bestCentToSplit is: ',bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]#当前中心（索引为bestCentToSplit）的坐标，换掉，成新的2中心的前一个的坐标（2维）
        centList.append(bestNewCents[1,:].tolist()[0])#再在最后追加新的2中心的后一个的坐标（2维）
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss#修改全局总ClusterAss，更新原来中心编号等于本轮bestCentToSplit的部分
    return mat(centList), clusterAssment

import urllib
import json
def testURLLib():
    #req = urllib.request.Request('http://120.131.82.100:18081/qa_child/child_app/?toyId=gqtest&t=hello')
    #c=urllib.request.urlopen(req)
    c=urllib.request.urlopen('http://120.131.82.100:18081/qa_child/child_app/?toyId=gqtest&t=hello')
    str = c.read().decode('utf-8')
    print(str)
    jobj =json.loads(str) 
    print(jobj)
    
def geoGrab(stAddress, city):
    apiStem = 'http://where.yahooapis.com/geocode?'  #create a dict and constants for the goecoder
    params = {}
    params['flags'] = 'J'#JSON return type
    params['appid'] = 'aaa0VN6k'
    params['location'] = '%s %s' % (stAddress, city)
    url_params = urllib.parse.urlencode(params)
    yahooApi = apiStem + url_params      #print url_params
    print(yahooApi)
    req = urllib.request.Request(yahooApi)
    c=urllib.request.urlopen(req)
    return json.loads(c.read())

from time import sleep
def massPlaceFind(fileName):
    fw = open('places.txt', 'w')
    for line in open(fileName).readlines():
        line = line.strip()
        lineArr = line.split('\t')
        retDict = geoGrab(lineArr[1], lineArr[2])
        if retDict['ResultSet']['Error'] == 0:
            lat = float(retDict['ResultSet']['Results'][0]['latitude'])
            lng = float(retDict['ResultSet']['Results'][0]['longitude'])
            print("%s\t%f\t%f" % (lineArr[0], lat, lng))
            fw.write('%s\t%f\t%f\n' % (line, lat, lng))
        else: print("error fetching")
        sleep(1)
    fw.close()
    
def distSLC(vecA, vecB):#Spherical Law of Cosines
    a = sin(vecA[0,1]*pi/180) * sin(vecB[0,1]*pi/180)
    b = cos(vecA[0,1]*pi/180) * cos(vecB[0,1]*pi/180) * \
                      cos(pi * (vecB[0,0]-vecA[0,0]) /180)
    return arccos(a + b)*6371.0 #pi is imported with numpy

import matplotlib
import matplotlib.pyplot as plt
def clusterClubs(numClust=5):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()
    rect=[0.1,0.1,0.8,0.8]
    scatterMarkers=['s', 'o', '^', '8', 'p', \
                    'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0=fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1=fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
    plt.show()
