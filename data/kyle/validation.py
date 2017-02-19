# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 19:18:03 2017

@author: kylem_000
"""

from time import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import scipy as sp
import pandas as pd
df = pd.read_csv("parsed.csv")
y1 = df["admission_type_id"].values
y2 = df["discharge_disposition_id"].values
columns = list(df)[1:4] + list(df)[8:49]
print columns
X = df[columns].values
      
X1_train, X1_test, y1_train, y1_test = train_test_split(X, y1, test_size = 0.20)
X2_train, X2_test, y2_train, y2_test = train_test_split(X, y2, test_size = 0.20)

clf1 = RandomForestClassifier()
clf2 = RandomForestClassifier()

clf1.fit(X1_train, y1_train)
clf2.fit(X2_train, y2_train)
y1_pred = clf1.predict(X1_test)
y2_pred = clf2.predict(X2_test)

acc1 = accuracy_score(y1_test, y1_pred)
acc2 = accuracy_score(y2_test, y2_pred)

print "accuracy1 mean:", acc1
print "accuracy2 mean:", acc2