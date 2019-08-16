#coding:utf-8
#import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import *

from numpy import *
import regTrees

class test(object):
    def __init__(self,parent):
        self.parent = parent
        fig = Figure(figsize=(5,4), dpi=100) #create canvas
        self.f = fig
        self.canvas = FigureCanvasTkAgg(fig, master=root)
        self.canvas.show()
        self.canvas.get_tk_widget().grid(row=0, columnspan=3)
        
        #Label(root, text="Plot Place Holder").grid(row=0, columnspan=3)
        Label(root, text="tolN").grid(row=1, column=0)
        self.tolNentry = Entry(self.parent)
        self.tolNentry.grid(row=1, column=1)
        self.tolNentry.insert(0,'10')
        Label(root, text="tolS").grid(row=2, column=0)
        self.tolSentry = Entry(root)
        self.tolSentry.grid(row=2, column=1)
        self.tolSentry.insert(0,'1.0')
        Button(root, text="ReDraw", command=self.drawNewTree).grid(row=1, column=2,rowspan=3)
        Button(root, text='Quit',fg="black", command=root.quit).grid(row=1,column=2)
        self.chkBtnVar = IntVar()
        self.chkBtn = Checkbutton(root, text="Model Tree", variable = self.chkBtnVar)
        self.chkBtn.grid(row=3, column=0, columnspan=2)
        self.rawDat = mat(regTrees.loadDataSet('sine.txt'))
        self.testDat = arange(min(self.rawDat[:,0]),max(self.rawDat[:,0]),0.01)
        self.reDraw(1.0, 10)
        
    def reDraw(self, tolS,tolN):
        self.f.clf()        # clear the figure
        self.a = self.f.add_subplot(111)
        if self.chkBtnVar.get():
            if tolN < 2: tolN = 2
            myTree=regTrees.createTree(self.rawDat, regTrees.modelLeaf,regTrees.modelErr, (tolS,tolN))
            yHat = regTrees.createForeCast(myTree, self.testDat,regTrees.modelTreeEval)
        else:
            myTree=regTrees.createTree(self.rawDat, ops=(tolS,tolN))
            yHat = regTrees.createForeCast(myTree, self.testDat)
        self.a.scatter(self.rawDat[:,0], self.rawDat[:,1], s=5) #use scatter for data set
        self.a.plot(self.testDat, yHat, linewidth=2.0) #use plot for yHat
        self.canvas.show()

    def getInputs(self):
        try: tolN = int(self.tolNentry.get())
        except: 
            tolN = 10 
            print("enter Integer for tolN")
            self.tolNentry.delete(0, END)
            self.tolNentry.insert(0,'10')
        try: tolS = float(self.tolSentry.get())
        except: 
            tolS = 1.0 
            print("enter Float for tolS")
            self.tolSentry.delete(0, END)
            self.tolSentry.insert(0,'1.0')
        return tolN,tolS

    def drawNewTree(self):
        tolN,tolS = self.getInputs()#get values from Entry boxes
        self.reDraw(tolS,tolN)

if __name__ == "__main__":
    root = Tk()
    test(root)
    root.mainloop()