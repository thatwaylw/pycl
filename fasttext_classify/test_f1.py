#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

from sklearn.metrics import f1_score
from sklearn import metrics

def get_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=[1], average='micro')

true_label = [0,1,0,1,1]
pred_label = [1,1,0,0,0]
f1_score = get_f1_score(true_label, pred_label)
print(f1_score)
p_score = metrics.precision_score(true_label, pred_label, labels=[1], average='macro')
print(p_score)
r_score = metrics.recall_score(true_label, pred_label, labels=[1], average='macro')
print(r_score)
