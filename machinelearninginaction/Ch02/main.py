'''
@author: laiwei
'''
import kNN
from numpy import *
#group, labels = kNN.createDataSet()
#print (group, labels)
#print (kNN.classify0([0, 0], group, labels, 3))
#print (kNN.classify0([0.7, 0.8], group, labels, 3))

datingMat, datingLabels = kNN.file2matrix("datingTestSet2.txt")
#print(datingMat)
#print(datingMat[:,0])
#print(datingLabels)

import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
#ax.scatter(datingMat[:,1], datingMat[:,2])
ax.scatter(datingMat[:,0], datingMat[:,1], 15.0*array(datingLabels), 15.0*array(datingLabels))
plt.show()


kNN.datingClassTest()
# kNN.handwritingClassTest()