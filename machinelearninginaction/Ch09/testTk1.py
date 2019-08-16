# -*- coding: utf-8 -*-
#import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import *

class test(object):
    def __init__(self,parent):
        self.parent = parent
        self.n = 0
        Button(self.parent,text = 'change',command = self.change).pack()
        '''表格个数'''        
        num = 2
        fig = []
        self.canvas = []
        for i in range(num):
            fig.append(Figure())
            self.canvas.append(FigureCanvasTkAgg(fig[i],master = self.parent))
        self.canvas[0]._tkcanvas.pack()
        '''以下可以创建不同的图'''
        axe = fig[0].add_subplot(111)
        #axe.set_title(u'第一个图',{'fontname':'STSong'})
        axe.set_title(u'first Fig')
        axe2 = fig[1].add_subplot(211)
        axe2_2 = fig[1].add_subplot(212)
        axe2.set_title(u'Fig.2')

    def change(self):
        self.n +=1
        if self.n == 2:
            self.n = 0
        self.canvas[self.n -1]._tkcanvas.pack_forget()
        self.canvas[self.n]._tkcanvas.pack()

if __name__ == "__main__":
    root = Tk()
    test(root)
    root.mainloop()