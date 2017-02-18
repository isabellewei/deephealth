# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 22:30:04 2017

@author: kylem_000
"""

from time import time
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import scipy as sp
import pandas as pd

df = pd.read_csv("parsed.csv")

y1 = df["admission_type_id"].values
y2 = df["discharge_disposition_id"].values
columns = list(df)[1:4] + list(df)[7:49]
X = df[columns].values

X_train, X_test, y1_train, y1_test = train_test_split(X, y1, test_size = 0.25)
X_train, X_test, y2_train, y2_test = train_test_split(X, y2, test_size = 0.25)      
            
clf1 = AdaBoostClassifier(n_estimators = 50)
clf2 = AdaBoostClassifier(n_estimators = 50)
clf1.fit(X, y1)
clf2.fit(X, y2)
y1_pred = clf1.predict(X_test)
y2_pred = clf2.predict(X_test)
acc1 = accuracy_score(y1_test, y1_pred)
acc2 = accuracy_score(y2_test, y2_pred)
print "accuracy1:", acc1
print "accuracy2:", acc2