# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd

s = pd.Series([1,2,3,np.nan,5,6])
print(s)

data = pd.read_csv('tmp/train.csv')     #读取csv文件
print(data.shape)
print(data)
print(data.dtypes)
print(data['content'])
# print(data[:,0])
# print(data[:,:])