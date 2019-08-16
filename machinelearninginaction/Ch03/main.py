'''
@author: laiwei
'''
import trees
myDat, labels=trees.createDataSet()
#myDat[0][-1]='maybe'
#print(trees.calcShannonEnt(myDat))
'''
print(myDat)
print(myDat[0])
print(len(myDat[0])-1)
'''
#ds1 = trees.splitDataSet(myDat,1,1)
#print(ds1)
#featList = [example[1] for example in myDat]
#print(featList)

#f1 = trees.chooseBestFeatureToSplit(myDat)
#print(f1)

'''
import operator
def dict2list(dic):
    keys = dic.keys()
    vals = dic.values()
    lst = [(key, val) for key, val in zip(keys, vals)]
    return lst
def testSort():
    classCount={}
    classCount['A'] = 2;
    classCount['B'] = 5;
    classCount['Cc'] = 1;
    #sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)  #python2.x
    sortedClassCount = sorted(dict2list(classCount), key=operator.itemgetter(1), reverse=True)    #python3.x  
    print(classCount)
    print(sortedClassCount)
    
    classList = [example[-1] for example in myDat]
    print(classList)
    mjCnt = trees.majorityCnt(classList)
    print(mjCnt)
testSort()
'''

myTree = trees.createTree(myDat, labels)
print(myTree)

import treePlotter
#treePlotter.createPlot0()

#myTree = treePlotter.retrieveTree(0)
#treePlotter.createPlot(myTree)
labels = ['no surfacing', 'flippers']
print(labels)
print(trees.classify(myTree, labels, [1,0]))
print(trees.classify(myTree, labels, [0,0]))
print(trees.classify(myTree, labels, [1,1]))
#trees.storeTree(myTree, 'myTree.txt')
#tree2 = trees.grabTree('myTree.txt')
fr=open('lenses.txt')
lenses=[inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels=['age','prescript', 'astigmatic', 'tearRate']
lensesTree = trees.createTree(lenses , lensesLabels)
print(lensesTree)
treePlotter.createPlot(lensesTree)