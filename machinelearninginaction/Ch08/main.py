#coding:utf-8
'''
Created on 2017年3月4日

@author: laiwei
'''
import regression

xArr,yArr = regression.loadDataSet('ex0.txt')

#ws = regression.standRegres(xArr, yArr)
#print(ws)
#regression.plot1(xArr, yArr, ws)

regression.plot3(xArr, yArr)

def test1():
    abX,abY=regression.loadDataSet('abalone.txt')
    yHat01=regression.lwlrTest(abX[0:99],abX[0:99],abY[0:99],0.1)
    yHat1=regression.lwlrTest(abX[0:99],abX[0:99],abY[0:99],1)
    yHat10=regression.lwlrTest(abX[0:99],abX[0:99],abY[0:99],10)
    print(regression.rssError(abY[0:99],yHat01.T))
    print(regression.rssError(abY[0:99],yHat1.T))
    print(regression.rssError(abY[0:99],yHat10.T))
    print('-------------------------------------------')
    yHat01=regression.lwlrTest(abX[100:199],abX[0:99],abY[0:99],0.1)
    yHat1=regression.lwlrTest(abX[100:199],abX[0:99],abY[0:99],1)
    yHat10=regression.lwlrTest(abX[100:199],abX[0:99],abY[0:99],10)
    print(regression.rssError(abY[100:199],yHat01.T))
    print(regression.rssError(abY[100:199],yHat1.T))
    print(regression.rssError(abY[100:199],yHat10.T))
    
#test1()

#regression.testMat0()
#regression.testMat()
regression.plot4()
#regression.testplotMat()
regression.testLasso()

##regression.scrapePage('setHtml/lego8288.html', '8288.txt', 2002, 800, 49.99)
#regression.test2()