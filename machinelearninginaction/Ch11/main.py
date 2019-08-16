#coding:utf-8
'''
Created on 2017年3月9日
@author: laiwei
'''
import apriori
from numpy import *

def test1():
    dataSet=apriori.loadDataSet()
    print(dataSet) #[[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
    
    #C1=apriori.createC1(dataSet)
    #print(set(C1)) #{frozenset({4}), frozenset({5}), frozenset({2}), frozenset({3}), frozenset({1})}
    #print(list(C1)) #[frozenset({1}), frozenset({2}), frozenset({3}), frozenset({4}), frozenset({5})]
    
    #D=map(set,dataSet)
    #print(list(D)) #[{1, 3, 4}, {2, 3, 5}, {1, 2, 3, 5}, {2, 5}] 注意！！被list(map1)之后，map1的内容就空了。。。好像set(.)也会清空人家
    
    #L1,suppData0 = apriori.scanD(D, C1, 0.5)  #不能直接用了，要把D和C1先变成list
    #print(L1)   #[frozenset({1}), frozenset({3}), frozenset({2}), frozenset({5})]
    #print(suppData0) #{frozenset({4}): 0.25, frozenset({5}): 0.75, frozenset({2}): 0.75, frozenset({3}): 0.75, frozenset({1}): 0.5}
    
    L,suppData = apriori.apriori(dataSet,0.5)
    print(L)
    print(suppData)
    rules = apriori.generateRules(L,suppData, minConf=0.5)
    print(rules)
    
#test1()

def test2():
    mushDatSet = [line.split() for line in open('mushroom.dat').readlines()]
    print(mat(mushDatSet[:5]))
    L, suppData = apriori.apriori(mushDatSet, minSupport=0.3)
    for item in L[2]:
        if item.intersection('2'): print(item)
        
test2()