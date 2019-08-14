#coding:utf-8
'''
Created on 2019年7月12日
@author: laiwei
'''
def insrt(hft, x):
    if(not hft): # == {}  #'v' in hft
        hft['v'] = x
        hft['l'] = {}
        hft['r'] = {}
        return
    if(x<hft['v']):
        insrt(hft['l'], x)
    else:
        insrt(hft['r'], x)

def trav(hft):          # 左序遍历
    if(not hft):    return
    trav(hft['l'])
    print(hft['v'])
    trav(hft['r'])
        
if __name__ == "__main__":

    hft = {}
    l = [4,7,2,5,9,5,7,6,3,1,8]
    for x in l:
        insrt(hft, x)
    
    print(hft)
    
    trav(hft)