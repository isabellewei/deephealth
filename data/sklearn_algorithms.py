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

train_df = pd.read_csv("parsed.csv")

columns = ["metformin", "repaglinide", "nateglinide", "chlorpropamide", 
"glimepiride", "acetohexamide", "glipizide", "glyburide", "tolbutamide", 
"pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone", 
"tolazamide", "examide", "citoglipton", "insulin", "glyburide-metformin",
"glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
"metformin-pioglitazone"]

y = train_df["admission_type_id"].values
X = train_df[list(columns)].values
y.astype(int)
X.astype(int)

             
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.80)
DT = DecisionTreeClassifier(max_depth = None, min_samples_split = 2, 
                            min_samples_leaf = 1)
clf = AdaBoostClassifier(base_estimator = DT, n_estimators = 50, 
                         learning_rate = 1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print "accuracy:", acc
