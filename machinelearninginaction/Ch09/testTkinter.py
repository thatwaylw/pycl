#coding:utf-8
'''
import os, sys
try:
    from tkinter import *
except ImportError:  #Python 2.x
    PythonVersion = 2
    from Tkinter import *
    from tkFont import Font
    from ttk import *
    from tkMessageBox import *
    import tkFileDialog
else:  #Python 3.x
    PythonVersion = 3
    from tkinter.font import Font
    from tkinter.ttk import *
    from tkinter.messagebox import *
    from tkinter import * 
'''

from tkinter import *
#from numpy import *
#import regTrees

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure                                #这两句顺序不能颠倒！！！
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg     #这两句顺序不能颠倒！！！

def reDraw(tolS,tolN):
    pass
def drawNewTree():
    pass

root=Tk()

reDraw.f = Figure(figsize=(5,4), dpi=100) #create canvas
reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
reDraw.canvas.show()
reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)

Label(root, text="Plot Place Holder").grid(row=0, columnspan=3)
Label(root, text="tolN").grid(row=1, column=0)
tolNentry = Entry(root)
tolNentry.grid(row=1, column=1)
tolNentry.insert(0,'10')
Label(root, text="tolS").grid(row=2, column=0)
tolSentry = Entry(root)
tolSentry.grid(row=2, column=1)
tolSentry.insert(0,'1.0')
Button(root, text="ReDraw", command=drawNewTree).grid(row=1, column=2,rowspan=3)
Button(root, text='Quit',fg="black", command=root.quit).grid(row=1,column=2)
chkBtnVar = IntVar()
chkBtn = Checkbutton(root, text="Model Tree", variable = chkBtnVar)
chkBtn.grid(row=3, column=0, columnspan=2)
#reDraw.rawDat = mat(regTrees.loadDataSet('sine.txt'))
#reDraw.testDat = arange(min(reDraw.rawDat[:,0]),max(reDraw.rawDat[:,0]),0.01)
reDraw(1.0, 10)
root.mainloop()