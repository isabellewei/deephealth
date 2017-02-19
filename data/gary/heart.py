from time import time
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
import  scipy as sp
import pandas as pd
import math

df = pd.read_csv("parsed_heart.csv")
y1 = df["num"].values

cols = list(df)
mlp = MLPClassifier(hidden_layer_sizes=(100,100,100))
clf1 = BaggingClassifier(n_estimators=10)
clf2 = BaggingClassifier(n_estimators=100)
clf3 = RandomForestClassifier(n_estimators=10,criterion='gini', min_samples_split=2,max_features=None)
clf4 = AdaBoostClassifier(n_estimators=100)
clf5 = VotingClassifier(estimators=[("rf",clf3),('bg',clf2),('ml',mlp),('ada',clf4)],voting='soft')


dropped = set(['num','id'])
columns2 = [z for z in cols if z not in dropped]
X2 = df[columns2].values
X_train, X_test, y_train, y_test = train_test_split(X2,y1,test_size=0.90)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp.fit(X_train,y_train)
predictions2 = mlp.predict(X_test)
print(classification_report(y_test, predictions2))
print(accuracy_score(y_test, predictions2))

kfold = KFold(n_splits=3,shuffle=True)
print(cross_val_score(mlp,X_test,y_test,cv=kfold).mean())

clf2.fit(X_train,y_train)
predictions = clf2.predict(X_test)
print(classification_report(y_test, predictions))
print(accuracy_score(y_test, predictions))

clf3.fit(X_train,y_train)
predictions2 = clf3.predict(X_test)
print(classification_report(y_test, predictions2))
print(accuracy_score(y_test, predictions2))

clf4.fit(X_train,y_train)
predictions2 = clf4.predict(X_test)
print(classification_report(y_test, predictions2))
print(accuracy_score(y_test, predictions2))

clf5.fit(X_train,y_train)
predictions2 = clf5.predict(X_test)
print(classification_report(y_test, predictions2))
print(accuracy_score(y_test, predictions2))

print(cross_val_score(clf5,X_test,y_test,cv=kfold).mean())




