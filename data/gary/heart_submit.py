from time import time
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
import  scipy as sp
import pandas as pd
import math

df = pd.read_csv('parsed_heart.csv')
df2 = pd.read_csv('parsed_heart_problem.csv')

cols_test = list(df2)
f_test = [x for x in cols_test if x not in set(['num','id'])]
f_test = df2[list(f_test)].values

labels = df['num'].values
cols = list(df)
f1 = [x for x in cols if x not in set(['num','id'])]
features1 = df[list(f1)].values

X_train, X_test, y_train, y_test = train_test_split(features1,labels,test_size=0.50)

mlp = MLPClassifier(hidden_layer_sizes=(100,100,100))
clf1 = BaggingClassifier(n_estimators=10)
clf2 = BaggingClassifier(n_estimators=100)
clf3 = RandomForestClassifier(n_estimators=10,criterion='gini', min_samples_split=2,max_features=None)
clf4 = AdaBoostClassifier(n_estimators=100)
clf5 = VotingClassifier(estimators=[("rf",clf3),('bg',clf2),('ml',mlp),('ada',clf4)],voting='soft')

scaler = StandardScaler()
scaler.fit(X_train)
f_test = scaler.transform(f_test)
X_train = scaler.transform(X_train)

clf5.fit(X_train, y_train)

pred1 = clf5.predict(f_test)
f_test = np.insert(f_test,6, pred1,1)
result = np.column_stack((df2['id'].values, pred1))
np.savetxt("output_heart_5.csv", result.astype(int), delimiter=",")
